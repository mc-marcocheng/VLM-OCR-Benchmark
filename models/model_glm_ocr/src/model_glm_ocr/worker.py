"""
Subprocess worker for GLM-OCR (transformers v5).

Two-stage pipeline matching the official GLM-OCR SDK:
  1. PP-DocLayoutV3  — detect regions with bboxes, categories, reading order
  2. GLM-OCR VLM     — recognise text in each cropped region

Protocol:
    python -m model_glm_ocr.worker --task task.json --output result.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

import psutil
import torch
from glmocr.config import LayoutConfig
from glmocr.layout import PPDocLayoutDetector, _layout_import_error
from glmocr.utils.image_utils import crop_image_region
from glmocr.utils.result_postprocess_utils import clean_repeated_content
from huggingface_hub import snapshot_download
from loguru import logger
from ocr_core.types import BBox, OCRPage, OCRRegion, WorkerTask
from ocr_core.utils import get_peak_vram_mb, get_vram_usage_mb, reset_peak_vram
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "zai-org/GLM-OCR"
LAYOUT_MODEL_ID = "PaddlePaddle/PP-DocLayoutV3_safetensors"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))
CACHE_DIR = os.path.join(_REPO_ROOT, "models", "huggingface_cache")

# Maps the layout detector's task_type → VLM prompt
# (these are the exact prompts the GLM-OCR model was trained with)
TASK_PROMPTS = {
    "text": "Text Recognition:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
}


# ── Helpers ──────────────────────────────────────────────────


def _get_ram() -> float:
    return psutil.Process().memory_info().rss / (1024**2)


# ── Model loading ───────────────────────────────────────────


def load_layout_detector(device: str = "cpu") -> PPDocLayoutDetector:
    """Load PP-DocLayoutV3 for layout detection.

    Runs on CPU by default so the GPU is fully available for the VLM.
    """
    if _layout_import_error is not None:
        raise ImportError(
            f"Layout detector unavailable: {_layout_import_error}"
        ) from _layout_import_error

    os.makedirs(CACHE_DIR, exist_ok=True)

    # Explicitly cache the layout model into the designated directory
    cached_model_dir = snapshot_download(repo_id=LAYOUT_MODEL_ID, cache_dir=CACHE_DIR)

    config = LayoutConfig(
        model_dir=cached_model_dir,
        device=device,
    )
    detector = PPDocLayoutDetector(config)
    detector.start()
    return detector


def load_vlm(device: str):
    """Load the GLM-OCR vision-language model."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    is_cuda = device.startswith("cuda")
    dtype = torch.float16 if is_cuda else torch.float32
    kwargs = {"torch_dtype": dtype, "cache_dir": CACHE_DIR}
    if is_cuda:
        kwargs["device_map"] = "auto"
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **kwargs)
    if not is_cuda:
        model.to(device)
    model.eval()
    return processor, model


# ── Inference ────────────────────────────────────────────────

import re

# Regex to strip ```markdown ... ``` wrappers the model produces
_MD_FENCE_RE = re.compile(
    r"^\s*```(?:markdown|html|latex|)?\s*\n?(.*?)\s*```\s*$",
    re.DOTALL,
)


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences the model wraps around its output."""
    m = _MD_FENCE_RE.match(text)
    return m.group(1).strip() if m else text.strip()


def recognize_region(
    processor,
    model,
    image: Image.Image,
    task_type: str,
    max_new_tokens: int = 8192,
) -> str:
    """Run the VLM on a single cropped region image."""
    prompt = TASK_PROMPTS.get(task_type, TASK_PROMPTS["text"])

    # ── Step 1: build text template (no tokenisation yet) ────
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},  # ← "image" not "content"
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,  # ← text only
        add_generation_prompt=True,
    )

    # ── Step 2: process text + image together ────────────────
    inputs = processor(
        text=[text_prompt],
        images=[image],  # ← images passed explicitly
        return_tensors="pt",
        padding=True,
    ).to(model.device)
    inputs.pop("token_type_ids", None)

    # ── Step 3: generate with SDK-matching parameters ────────
    with torch.inference_mode():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,  # ← greedy, matching SDK
            repetition_penalty=1.1,  # ← matching SDK
            do_sample=False,  # ← deterministic
        )

    new_tokens = gen[0][inputs["input_ids"].shape[1] :]
    raw = processor.decode(new_tokens, skip_special_tokens=True)

    # ── Step 4: strip markdown fences + hallucination guard ──
    text = _strip_markdown_fences(raw)

    cleaned = clean_repeated_content(text)
    return cleaned if cleaned else text


def predict(
    processor,
    model,
    layout_detector: PPDocLayoutDetector,
    image_path: str,
    max_new_tokens: int = 8192,
) -> OCRPage:
    """Full two-stage prediction: layout → per-region VLM recognition."""
    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size

    # ── Stage 1: layout detection ────────────────────────────
    layout_results, _vis = layout_detector.process(
        [image],
        save_visualization=False,
    )
    regions_raw = layout_results[0] if layout_results else []
    logger.info(
        f"{os.path.basename(image_path)}: layout detected {len(regions_raw)} regions"
    )

    # ── Stage 2: per-region recognition ──────────────────────
    ocr_regions: list[OCRRegion] = []
    text_parts: list[str] = []

    for region in regions_raw:
        task_type = region.get("task_type", "text")
        label = region.get("label", "text")
        bbox_2d = region.get("bbox_2d", [0, 0, 1000, 1000])
        order = region.get("index", len(ocr_regions))

        # Normalised 0-1000 → pixel coordinates
        bbox = BBox(
            x1=bbox_2d[0] / 1000.0 * img_w,
            y1=bbox_2d[1] / 1000.0 * img_h,
            x2=bbox_2d[2] / 1000.0 * img_w,
            y2=bbox_2d[3] / 1000.0 * img_h,
        )

        # "abandon" = header/footer/page-number — discard entirely
        if task_type == "abandon":
            continue

        # "skip" = image/chart — keep bbox but no text recognition
        if task_type == "skip":
            ocr_regions.append(
                OCRRegion(
                    text="",
                    category=label,
                    bbox=bbox,
                    order=order,
                )
            )
            continue

        # Crop region and recognise with VLM
        try:
            polygon = region.get("polygon")
            cropped = crop_image_region(image, bbox_2d, polygon)
            text = recognize_region(
                processor,
                model,
                cropped,
                task_type,
                max_new_tokens,
            )
        except Exception as e:
            logger.warning(f"Recognition failed for region {order} ({label}): {e}")
            text = ""

        ocr_regions.append(
            OCRRegion(
                text=text,
                category=label,
                bbox=bbox,
                order=order,
            )
        )
        if text:
            text_parts.append(text)

    full_text = "\n".join(text_parts)
    return OCRPage(
        full_text=full_text,
        width=img_w,
        height=img_h,
        regions=ocr_regions,
    )


# ── CLI entry-point ─────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.task, "r") as f:
        task = WorkerTask.from_dict(json.load(f))

    ram_before = _get_ram()
    vram_before = get_vram_usage_mb()
    reset_peak_vram()

    t0 = time.perf_counter()

    # Layout detector on CPU — saves VRAM for the VLM
    layout_detector = load_layout_detector(device="cpu")

    # VLM on the requested device (cuda / cpu)
    processor, model = load_vlm(task.device)
    load_time = time.perf_counter() - t0

    max_new_tokens = task.params.get("max_new_tokens", 8192)
    ram_after = _get_ram()
    vram_after = get_vram_usage_mb()
    peak_ram = ram_after

    pages = []
    for img_path in task.image_paths:
        t1 = time.perf_counter()
        error = None
        try:
            result = predict(
                processor,
                model,
                layout_detector,
                img_path,
                max_new_tokens=max_new_tokens,
            )
        except Exception as e:
            logger.exception(f"Failed on {img_path}")
            error = str(e)
            result = OCRPage(full_text="")
        elapsed = time.perf_counter() - t1
        cur_ram = _get_ram()
        peak_ram = max(peak_ram, cur_ram)
        pages.append(
            {
                "image_path": img_path,
                "prediction_time_seconds": round(elapsed, 4),
                "ram_after_mb": round(cur_ram, 2),
                "result": result.to_dict(),
                "error": error,
            }
        )

    layout_detector.stop()

    output = {
        "model_load_time_seconds": round(load_time, 4),
        "ram_before_load_mb": round(ram_before, 2),
        "ram_after_load_mb": round(ram_after, 2),
        "peak_ram_mb": round(peak_ram, 2),
        "vram_before_load_mb": (
            round(vram_before, 2) if vram_before is not None else None
        ),
        "vram_after_load_mb": round(vram_after, 2) if vram_after is not None else None,
        "peak_vram_mb": (
            round(get_peak_vram_mb(), 2) if get_peak_vram_mb() is not None else None
        ),
        "pages": pages,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"Worker done → {args.output}")


if __name__ == "__main__":
    main()
