"""PaddleOCR-VL worker implementing the subprocess JSON protocol."""

from __future__ import annotations

import argparse
import json
import os
import time

# ── Configure HuggingFace Cache ──────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))
CACHE_DIR = os.path.join(_REPO_ROOT, "models", "huggingface_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Set environment variables BEFORE importing transformers
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_HUB_CACHE"] = CACHE_DIR
# ─────────────────────────────────────────────────────────────

import psutil
import torch
from loguru import logger
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from ocr_core.types import OCRPage, WorkerPageResult, WorkerResponse, WorkerTask
from ocr_core.utils import (
    get_peak_vram_mb,
    get_vram_usage_mb,
    reset_peak_vram,
    resolve_device,
)

# ── Constants ────────────────────────────────────────────────
MODEL_PATH = "PaddlePaddle/PaddleOCR-VL-1.5"

PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
    "spotting": "Spotting:",
    "seal": "Seal Recognition:",
}

SPOTTING_UPSCALE_THRESHOLD = 1500
DEFAULT_MAX_NEW_TOKENS = 8192
DEFAULT_MIN_PIXELS = 256 * 28 * 28


def preprocess_image(image: Image.Image, task: str) -> tuple[Image.Image, int]:
    """
    Preprocess image for the given task.

    Returns the processed image and the max_pixels setting.
    """
    orig_w, orig_h = image.size

    # Special upscaling for spotting task on small images
    if (
        task == "spotting"
        and orig_w < SPOTTING_UPSCALE_THRESHOLD
        and orig_h < SPOTTING_UPSCALE_THRESHOLD
    ):
        process_w, process_h = orig_w * 2, orig_h * 2
        try:
            resample_filter = Image.Resampling.LANCZOS
        except AttributeError:
            resample_filter = Image.LANCZOS
        image = image.resize((process_w, process_h), resample_filter)

    # Set max_pixels based on task
    if task == "spotting":
        max_pixels = 2048 * 28 * 28  # 1605632
    else:
        max_pixels = 1280 * 28 * 28  # ~1M pixels

    return image, max_pixels


def run_inference(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    image: Image.Image,
    task: str,
    max_new_tokens: int,
) -> str:
    """Run OCR inference on a single image."""
    # Preprocess
    processed_image, max_pixels = preprocess_image(image, task)

    # Get min_pixels from processor if available, otherwise use default
    min_pixels = getattr(processor.image_processor, "min_pixels", DEFAULT_MIN_PIXELS)

    # Build messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": processed_image},
                {"type": "text", "text": PROMPTS.get(task, PROMPTS["ocr"])},
            ],
        }
    ]

    # Tokenize
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        images_kwargs={
            "size": {
                "shortest_edge": min_pixels,
                "longest_edge": max_pixels,
            }
        },
    ).to(model.device)

    # Generate
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Decode (skip input tokens and final token)
    input_len = inputs["input_ids"].shape[-1]
    result = processor.decode(outputs[0][input_len:-1])

    return result


def load_model(
    device: str, use_flash_attn: bool = True
) -> tuple[AutoModelForImageTextToText, AutoProcessor]:
    """Load model and processor."""
    logger.info(f"Loading PaddleOCR-VL from {MODEL_PATH}")

    # Determine attention implementation
    attn_impl = None
    if use_flash_attn and device.startswith("cuda"):
        try:
            import flash_attn  # noqa: F401

            attn_impl = "flash_attention_2"
            logger.info("Using flash_attention_2")
        except ImportError:
            logger.warning("flash-attn not installed, using default attention")

    # Load model
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "cache_dir": CACHE_DIR,
    }
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, **model_kwargs)
    model = model.to(device).eval()

    # Load processor
    processor = AutoProcessor.from_pretrained(MODEL_PATH, cache_dir=CACHE_DIR)

    return model, processor


def main() -> None:
    """Main entry point for the worker."""
    parser = argparse.ArgumentParser(description="PaddleOCR-VL worker")
    parser.add_argument("--task", required=True, help="Path to input task JSON")
    parser.add_argument("--output", required=True, help="Path to output response JSON")
    args = parser.parse_args()

    # Parse input task
    with open(args.task, "r", encoding="utf-8") as f:
        task = WorkerTask.from_dict(json.load(f))

    device = resolve_device(task.device)
    logger.info(f"PaddleOCR-VL worker starting on {device}")

    # Extract model params
    params = task.params or {}
    ocr_task = params.get("task", "ocr")
    max_new_tokens = params.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
    use_flash_attn = params.get("use_flash_attn", True)

    if ocr_task not in PROMPTS:
        logger.warning(f"Unknown task '{ocr_task}', falling back to 'ocr'")
        ocr_task = "ocr"

    logger.info(f"OCR task: {ocr_task}, max_new_tokens: {max_new_tokens}")

    # 1. Track Load Time & Initial Resources
    t0 = time.perf_counter()
    ram_before = psutil.Process().memory_info().rss / (1024 * 1024)
    vram_before = get_vram_usage_mb()

    # Load model
    model, processor = load_model(device, use_flash_attn=use_flash_attn)

    load_time = time.perf_counter() - t0
    ram_after = psutil.Process().memory_info().rss / (1024 * 1024)
    vram_after = get_vram_usage_mb()

    logger.info(f"Model loaded in {load_time:.2f}s")
    logger.info(f"VRAM: {vram_before:.0f}MB -> {vram_after:.0f}MB")

    reset_peak_vram()

    # 2. Process Pages
    pages: list[WorkerPageResult] = []
    for img_path in task.image_paths:
        logger.info(f"Processing {img_path}")
        t_start = time.perf_counter()

        try:
            # Load image
            image = Image.open(img_path).convert("RGB")

            # Run inference
            text = run_inference(model, processor, image, ocr_task, max_new_tokens)

            # Create result
            result_page = OCRPage(full_text=text.strip(), regions=[])
            pred_time = time.perf_counter() - t_start

            pages.append(
                WorkerPageResult(
                    image_path=img_path,
                    prediction_time_seconds=pred_time,
                    result=result_page,
                )
            )
            logger.info(f"  Completed in {pred_time:.2f}s, {len(text)} chars")

        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            pages.append(
                WorkerPageResult(
                    image_path=img_path,
                    prediction_time_seconds=time.perf_counter() - t_start,
                    error=str(e),
                    result=OCRPage(full_text="", regions=[]),
                )
            )

    # 3. Finalize Resources
    peak_ram = psutil.Process().memory_info().rss / (1024 * 1024)
    peak_vram = get_peak_vram_mb()

    # 4. Serialize Output
    response = WorkerResponse(
        model_load_time_seconds=load_time,
        ram_before_load_mb=ram_before,
        ram_after_load_mb=ram_after,
        peak_ram_mb=peak_ram,
        vram_before_load_mb=vram_before,
        vram_after_load_mb=vram_after,
        peak_vram_mb=peak_vram,
        pages=pages,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(response.to_dict(), f, ensure_ascii=False, indent=2)

    logger.info(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
