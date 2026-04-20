"""
Subprocess worker for MonkeyOCR.

MonkeyOCR combines layout detection with a finetuned **Qwen2.5-VL** model
that performs per-region OCR, table-to-HTML, and formula-to-LaTeX extraction.

This worker reimplements the ``MonkeyChat_transformers`` inference path from
the official MonkeyOCR repo directly, avoiding the heavy ``magic_pdf`` /
PaddlePaddle / PaddleX dependency chain.  Layout detection is handled by
**DocLayout-YOLO** (lighter alternative to PaddleX PP-DocLayoutV2).

Protocol
--------
    python -m model_monkey_ocr --task task.json --output result.json

Required ``params`` in task JSON
--------------------------------
    model_id : str
        HuggingFace repo ID for MonkeyOCR.
        Available: ``"echo840/MonkeyOCR-pro-1.2B"`` (fast) or
        ``"echo840/MonkeyOCR-pro-3B"`` (accurate).

Optional ``params``
-------------------
    layout_model_repo : str
        HF repo for DocLayout-YOLO weights.
        Default ``"juliozhao/DocLayout-YOLO-DocStructBench"``.
    layout_model_file : str
        Filename inside that repo.
        Default ``"doclayout_yolo_docstructbench_imgsz1024.pt"``.
    batch_size : int   - max batch size for the chat model  (default 8)
    max_image_size : int - longest-edge pixel cap            (default 1600)
    layout_conf : float - DocLayout-YOLO confidence thresh   (default 0.25)
    layout_imgsz : int  - DocLayout-YOLO input size          (default 1024)
    use_layout : bool   - set False to skip layout entirely  (default False)
    layout_iou_thresh : float - NMS IoU threshold for dedup  (default 0.5)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import psutil

# ── Configure caches BEFORE any model library imports ────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))
CACHE_DIR = os.path.join(_REPO_ROOT, "models", "huggingface_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

os.environ.setdefault("HF_HOME", CACHE_DIR)
os.environ.setdefault("HF_HUB_CACHE", CACHE_DIR)
# ──────────────────────────────────────────────────────────────────

import torch
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download, snapshot_download
from loguru import logger
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from ocr_core.types import BBox, OCRPage, OCRRegion, WorkerPageResult, WorkerTask
from ocr_core.utils import get_peak_vram_mb, get_vram_usage_mb, reset_peak_vram

# ── Constants ─────────────────────────────────────────────────────

# DocLayout-YOLO class-id → (ocr-core category, task type)
# task type: "text" | "table" | "formula" | "skip"
LAYOUT_CLASSES: dict[int, tuple[str, str]] = {
    0: ("title", "text"),
    1: ("text", "text"),
    2: ("abandon", "skip"),
    3: ("picture", "skip"),
    4: ("caption", "text"),
    5: ("table", "table"),
    6: ("caption", "text"),
    7: ("footnote", "text"),
    8: ("formula", "formula"),
    9: ("caption", "text"),
}

# Per-task instructions matching MonkeyOCR's prompts
TASK_INSTRUCTIONS: dict[str, str] = {
    "text": "Please output the text content from the image.",
    "table": (
        "This is the image of a table. " "Please output the table in html format."
    ),
    "formula": (
        "Please write out the expression of the formula "
        "in the image using LaTeX format."
    ),
    "full_page": (
        "Please recognize and output all the text content "
        "from this image. Output the text exactly as it appears, "
        "preserving the original language, numbers, and symbols."
    ),
}


def get_ram_mb() -> float:
    return psutil.Process().memory_info().rss / (1024**2)


# ── Image helpers ─────────────────────────────────────────────────


def resize_image(image: Image.Image, max_size: int = 1600) -> Image.Image:
    """Resize *image* so its longest edge is at most *max_size* pixels."""
    image = image.convert("RGB")
    w, h = image.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        image = image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    return image


def crop_region(image: Image.Image, bbox: list[float], padding: int = 2) -> Image.Image:
    """Crop a region from *image* with a small pixel padding."""
    w, h = image.size
    x1 = max(0, int(bbox[0]) - padding)
    y1 = max(0, int(bbox[1]) - padding)
    x2 = min(w, int(bbox[2]) + padding)
    y2 = min(h, int(bbox[3]) + padding)
    return image.crop((x1, y1, x2, y2))


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ── Model downloading ────────────────────────────────────────────


def download_monkeyocr_repo(repo_id: str) -> str:
    """
    Download the full MonkeyOCR repo (contains Recognition/, Relation/,
    Structure/ sub-models) and return the local directory path.

    Repo structure on HuggingFace::

        echo840/MonkeyOCR-pro-1.2B/
        ├── Recognition/       ← Qwen2.5-VL chat model
        │   ├── config.json
        │   ├── model*.safetensors
        │   ├── tokenizer.json
        │   └── ...
        ├── Relation/          ← LayoutReader (LayoutLMv3)
        │   ├── config.json
        │   └── model.safetensors
        └── Structure/         ← PP-DocLayoutV2 (PaddleX, unused by us)
    """
    local_dir = os.path.join(CACHE_DIR, "monkeyocr_repos", repo_id.replace("/", "--"))

    # Check if Recognition model already exists
    recognition_dir = os.path.join(local_dir, "Recognition")
    if os.path.isdir(recognition_dir) and os.path.isfile(
        os.path.join(recognition_dir, "config.json")
    ):
        logger.info(f"MonkeyOCR repo already downloaded at {local_dir}")
        return local_dir

    logger.info(f"Downloading MonkeyOCR repo: {repo_id}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        # Skip PaddleX structure files (we use DocLayout-YOLO instead)
        ignore_patterns=["Structure/**"],
    )
    return local_dir


# ── DocLayout-YOLO wrapper ───────────────────────────────────────


class LayoutDetector:
    """Thin wrapper around DocLayout-YOLO with NMS deduplication."""

    def __init__(
        self,
        repo_id: str,
        filename: str,
        device: str,
    ):
        weight_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=CACHE_DIR,
        )
        self.model = YOLOv10(weight_path)
        self.device = device

    def detect(
        self,
        image: Image.Image,
        conf: float = 0.25,
        imgsz: int = 1024,
        iou_thresh: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Return deduplicated detections sorted top-to-bottom."""
        results = self.model.predict(
            image,
            imgsz=imgsz,
            conf=conf,
            device=self.device,
            verbose=False,
        )
        raw: list[dict[str, Any]] = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().tolist()
                cls_id = int(boxes.cls[i].cpu().item())
                score = float(boxes.conf[i].cpu().item())
                raw.append({"bbox": xyxy, "cls": cls_id, "conf": score})

        # Sort by confidence descending for greedy NMS
        raw.sort(key=lambda d: d["conf"], reverse=True)

        # Greedy dedup: suppress lower-confidence boxes that overlap
        kept: list[dict[str, Any]] = []
        for det in raw:
            duplicate = False
            for existing in kept:
                if compute_iou(det["bbox"], existing["bbox"]) > iou_thresh:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(det)

        # Sort top-to-bottom, then left-to-right for reading order
        kept.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))
        return kept


# ── Qwen2.5-VL chat model (mirrors MonkeyChat_transformers) ─────


class ChatModel:
    """
    Loads MonkeyOCR's finetuned Qwen2.5-VL model and exposes
    ``batch_inference``.  Mirrors the ``MonkeyChat_transformers`` class
    from the official repo.
    """

    def __init__(
        self,
        model_path: str,
        max_batch_size: int = 8,
        max_new_tokens: int = 4096,
        device: str = "cuda",
    ):
        self.device = device
        self.max_batch_size = max_batch_size
        self.max_new_tokens = max_new_tokens

        bf16 = False
        if device.startswith("cuda"):
            bf16 = torch.cuda.is_bf16_supported()
        elif device.startswith("mps"):
            bf16 = True
        dtype = torch.bfloat16 if bf16 else torch.float16

        attn_impl = "flash_attention_2" if device.startswith("cuda") else "sdpa"
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=dtype,
                attn_implementation=attn_impl,
                device_map=device,
            )
        except Exception:
            logger.warning("flash_attention_2 unavailable, falling back to sdpa")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=dtype,
                attn_implementation="sdpa",
                device_map=device,
            )

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.processor.tokenizer.padding_side = "left"
        self.model.eval()
        logger.info(f"Qwen2.5-VL loaded from {model_path} (dtype={dtype})")

    # ---- public API -------------------------------------------------

    def batch_inference(
        self,
        images: list[Image.Image],
        questions: list[str],
    ) -> list[str]:
        """Run batched VL inference; one text string per image."""
        assert len(images) == len(questions)
        results: list[str] = []

        for start in range(0, len(images), self.max_batch_size):
            end = min(start + self.max_batch_size, len(images))
            batch_imgs = images[start:end]
            batch_qs = questions[start:end]

            try:
                texts_out = self._forward_batch(
                    batch_imgs, batch_qs, process_vision_info
                )
                results.extend(texts_out)
            except Exception as exc:
                logger.warning(f"Batch failed ({exc}), falling back to singles")
                for img, q in zip(batch_imgs, batch_qs):
                    try:
                        texts_out = self._forward_batch([img], [q], process_vision_info)
                        results.extend(texts_out)
                    except Exception as single_exc:
                        logger.error(f"Single inference failed: {single_exc}")
                        results.append("")

            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()

        return results

    # ---- internals --------------------------------------------------

    @torch.no_grad()
    def _forward_batch(
        self,
        images: list[Image.Image],
        questions: list[str],
        process_vision_info,
    ) -> list[str]:
        all_messages = []
        for img, q in zip(images, questions):
            all_messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": q},
                        ],
                    }
                ]
            )

        texts: list[str] = []
        image_inputs_list: list = []
        for msgs in all_messages:
            text = self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)
            img_in, _ = process_vision_info(msgs)
            image_inputs_list.append(img_in)

        inputs = self.processor(
            text=texts,
            images=image_inputs_list,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.1,
            repetition_penalty=1.05,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )

        trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]
        decoded = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [t.strip() for t in decoded]


# ── Pipeline: layout → per-region OCR → OCRPage ─────────────────


def _text_format_for_task(task_type: str) -> str:
    if task_type == "table":
        return "html"
    if task_type == "formula":
        return "latex"
    return "plain"


def process_image_with_layout(
    image: Image.Image,
    chat: ChatModel,
    layout: LayoutDetector,
    max_image_size: int,
    layout_conf: float,
    layout_imgsz: int,
    layout_iou_thresh: float,
) -> OCRPage:
    """Full pipeline: detect layout → per-region recognition."""
    orig_w, orig_h = image.size
    resized = resize_image(image, max_image_size)

    detections = layout.detect(
        resized,
        conf=layout_conf,
        imgsz=layout_imgsz,
        iou_thresh=layout_iou_thresh,
    )

    if not detections:
        return process_image_full_page(image, chat, max_image_size)

    res_w, res_h = resized.size

    crop_images: list[Image.Image] = []
    crop_indices: list[int] = []
    crop_tasks: list[str] = []
    region_data: list[dict[str, Any]] = []

    for idx, det in enumerate(detections):
        cls_id = det["cls"]
        info = LAYOUT_CLASSES.get(cls_id, ("text", "text"))
        category, task_type = info

        bbox_res = det["bbox"]
        sx = orig_w / res_w
        sy = orig_h / res_h
        bbox_orig = [
            bbox_res[0] * sx,
            bbox_res[1] * sy,
            bbox_res[2] * sx,
            bbox_res[3] * sy,
        ]

        region_data.append(
            {
                "bbox_orig": bbox_orig,
                "bbox_res": bbox_res,
                "category": category,
                "task_type": task_type,
                "confidence": det["conf"],
                "text": "",
            }
        )

        if task_type != "skip":
            cropped = crop_region(resized, bbox_res)
            cw, ch = cropped.size
            if cw < 4 or ch < 4:
                continue
            crop_images.append(cropped)
            crop_indices.append(idx)
            crop_tasks.append(task_type)

    if crop_images:
        instructions = [TASK_INSTRUCTIONS[t] for t in crop_tasks]
        responses = chat.batch_inference(crop_images, instructions)
        for ci, response in zip(crop_indices, responses):
            region_data[ci]["text"] = response

    regions: list[OCRRegion] = []
    full_texts: list[str] = []

    for order, rd in enumerate(region_data):
        bx = rd["bbox_orig"]
        bbox = BBox(
            x1=min(bx[0], bx[2]),
            y1=min(bx[1], bx[3]),
            x2=max(bx[0], bx[2]),
            y2=max(bx[1], bx[3]),
        )
        text = rd["text"]
        category = rd["category"]
        task_type = rd["task_type"]

        region = OCRRegion(
            text=text,
            category=category,
            bbox=bbox,
            order=order,
            text_format=_text_format_for_task(task_type),
            confidence=rd["confidence"],
        )
        regions.append(region)

        if text and task_type != "skip":
            full_texts.append(text)

    return OCRPage(
        full_text="\n\n".join(full_texts),
        regions=regions,
        width=orig_w,
        height=orig_h,
    )


def process_image_full_page(
    image: Image.Image,
    chat: ChatModel,
    max_image_size: int,
) -> OCRPage:
    """Send entire image to the chat model for OCR."""
    orig_w, orig_h = image.size
    resized = resize_image(image, max_image_size)

    responses = chat.batch_inference([resized], [TASK_INSTRUCTIONS["full_page"]])
    text = responses[0] if responses else ""

    return OCRPage(
        full_text=text,
        regions=[],
        width=orig_w,
        height=orig_h,
    )


# ── Main ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="MonkeyOCR worker")
    parser.add_argument("--task", required=True, help="Path to task JSON")
    parser.add_argument("--output", required=True, help="Path for output JSON")
    args = parser.parse_args()

    with open(args.task, "r", encoding="utf-8") as f:
        task = WorkerTask.from_dict(json.load(f))

    # ── Resolve parameters ────────────────────────────────────────
    params = task.params
    model_id = params.get("model_id", "echo840/MonkeyOCR-pro-1.2B")

    device = task.device or "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        logger.warning("CUDA not available, falling back to CPU")

    batch_size = int(params.get("batch_size", 8))
    max_image_size = int(params.get("max_image_size", 1600))
    use_layout = params.get("use_layout", False)
    layout_repo = params.get(
        "layout_model_repo",
        "juliozhao/DocLayout-YOLO-DocStructBench",
    )
    layout_file = params.get(
        "layout_model_file",
        "doclayout_yolo_docstructbench_imgsz1024.pt",
    )
    layout_conf = float(params.get("layout_conf", 0.25))
    layout_imgsz = int(params.get("layout_imgsz", 1024))
    layout_iou_thresh = float(params.get("layout_iou_thresh", 0.5))

    # ── Memory tracking — before load ─────────────────────────────
    ram_before = get_ram_mb()
    vram_before = get_vram_usage_mb()
    reset_peak_vram()

    t0 = time.perf_counter()

    # ── Download MonkeyOCR repo & resolve Recognition model path ──
    if os.path.isdir(model_id) and os.path.isfile(
        os.path.join(model_id, "config.json")
    ):
        recognition_path = model_id
    elif os.path.isdir(model_id) and os.path.isdir(
        os.path.join(model_id, "Recognition")
    ):
        recognition_path = os.path.join(model_id, "Recognition")
    else:
        repo_dir = download_monkeyocr_repo(model_id)
        recognition_path = os.path.join(repo_dir, "Recognition")

    if not os.path.isdir(recognition_path):
        raise FileNotFoundError(
            f"Recognition model not found at '{recognition_path}'. "
            f"Repo ID was: {model_id}"
        )

    logger.info(f"Recognition model path: {recognition_path}")

    # ── Load chat model ───────────────────────────────────────────
    chat = ChatModel(
        model_path=recognition_path,
        max_batch_size=batch_size,
        device=device,
    )

    # ── Load layout model (optional) ──────────────────────────────
    layout: LayoutDetector | None = None
    if use_layout:
        try:
            logger.info(f"Loading DocLayout-YOLO: {layout_repo}/{layout_file}")
            layout = LayoutDetector(
                repo_id=layout_repo,
                filename=layout_file,
                device=device,
            )
            logger.info("DocLayout-YOLO loaded")
        except Exception as exc:
            logger.warning(
                f"DocLayout-YOLO unavailable ({exc}); " "using full-page OCR"
            )
            layout = None

    load_time = time.perf_counter() - t0
    ram_after = get_ram_mb()
    vram_after = get_vram_usage_mb()
    logger.info(f"Models loaded in {load_time:.2f}s")

    # ── Process images ────────────────────────────────────────────
    pages: list[dict] = []
    for img_path in task.image_paths:
        logger.info(f"Processing: {img_path}")
        try:
            image = Image.open(img_path).convert("RGB")
            t1 = time.perf_counter()

            if layout is not None:
                ocr_page = process_image_with_layout(
                    image,
                    chat,
                    layout,
                    max_image_size,
                    layout_conf,
                    layout_imgsz,
                    layout_iou_thresh,
                )
            else:
                ocr_page = process_image_full_page(image, chat, max_image_size)

            pred_time = time.perf_counter() - t1
            ocr_page.page_number = len(pages) + 1

            pages.append(
                WorkerPageResult(
                    image_path=img_path,
                    prediction_time_seconds=pred_time,
                    ram_after_mb=get_ram_mb(),
                    result=ocr_page,
                ).to_dict()
            )
            logger.info(
                f"  → {len(ocr_page.regions)} regions, "
                f"{len(ocr_page.full_text)} chars, {pred_time:.2f}s"
            )

        except Exception:
            logger.exception(f"Failed to process {img_path}")
            pages.append(
                WorkerPageResult(
                    image_path=img_path,
                    error="processing failed",
                    result=OCRPage(regions=[]),
                ).to_dict()
            )

    # ── Write output ──────────────────────────────────────────────
    output = {
        "model_load_time_seconds": load_time,
        "ram_before_load_mb": ram_before,
        "ram_after_load_mb": ram_after,
        "peak_ram_mb": get_ram_mb(),
        "vram_before_load_mb": vram_before,
        "vram_after_load_mb": vram_after,
        "peak_vram_mb": get_peak_vram_mb(),
        "pages": pages,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"Worker done → {args.output}")


if __name__ == "__main__":
    main()
