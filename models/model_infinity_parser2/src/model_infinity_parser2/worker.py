"""
Subprocess worker for Infinity-Parser2.

Loads an Infinity-Parser2 model (a fine-tuned Qwen2.5-VL variant)
and runs structured document OCR inference. The model outputs a JSON array of
layout elements with bounding boxes (normalised to [0-1000]), categories, and
text content formatted per category (HTML for tables, LaTeX for formulas,
Markdown for everything else).

Protocol:
    python -m model_infinity_parser2.worker --task task.json --output result.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import psutil
import torch
from loguru import logger
from PIL import Image

# ── Configure HuggingFace Cache ──────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))
CACHE_DIR = os.path.join(_REPO_ROOT, "models", "huggingface_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_HUB_CACHE"] = CACHE_DIR
# ─────────────────────────────────────────────────────────────

from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

from model_infinity_parser2.utils import PROMPT_DOC2JSON, postprocess_doc2json_result
from ocr_core.types import BBox, OCRPage, OCRRegion, WorkerPageResult, WorkerTask
from ocr_core.utils import get_peak_vram_mb, get_vram_usage_mb, reset_peak_vram

MODEL_ID = "infly/Infinity-Parser2-Pro"
DEFAULT_MIN_PIXELS = 2048  # 32 × 64
DEFAULT_MAX_PIXELS = 16_777_216  # 4096 × 4096
DEFAULT_MAX_NEW_TOKENS = 32768

# Category mapping from model output to ocr-core standard labels
CATEGORY_MAP: dict[str, str] = {
    "header": "page-header",
    "title": "title",
    "text": "text",
    "figure": "picture",
    "table": "table",
    "formula": "formula",
    "figure_caption": "caption",
    "table_caption": "caption",
    "formula_caption": "caption",
    "figure_footnote": "footnote",
    "table_footnote": "footnote",
    "page_footnote": "footnote",
    "footer": "page-footer",
}

# Map categories to their expected text format
TEXT_FORMAT_MAP: dict[str, str] = {
    "table": "html",
    "formula": "latex",
    "picture": "plain",
}


def get_ram_mb() -> float:
    """Get current process RAM usage in MB."""
    return psutil.Process().memory_info().rss / (1024**2)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class InfinityParser2Model:
    """Wrapper around Infinity-Parser2 VLM for structured document OCR."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = "cuda",
        torch_dtype: str = "float16",
        min_pixels: int = DEFAULT_MIN_PIXELS,
        max_pixels: int = DEFAULT_MAX_PIXELS,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype, torch.float16)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.model = None
        self.processor = None

    def load(self) -> None:
        """Load model weights and processor from HuggingFace."""
        logger.info(f"Loading Infinity-Parser2 from {self.model_id}")

        # Determine attention implementation
        if self.device == "cuda":
            try:
                import flash_attn  # noqa: F401

                attn_impl = "flash_attention_2"
                logger.info("Using flash_attention_2 backend")
            except ImportError:
                attn_impl = "sdpa"
                logger.warning("flash_attn not available, falling back to SDPA")
        else:
            attn_impl = "eager"

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
            attn_implementation=attn_impl,
            cache_dir=CACHE_DIR,
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            cache_dir=CACHE_DIR,
        )

        logger.info("Infinity-Parser2 loaded successfully")

    def predict(
        self,
        image: Image.Image,
        prompt: str = PROMPT_DOC2JSON,
        *,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        """Run structured layout OCR inference on a single page.

        Args:
            image: Document page as a PIL Image (RGB).
            prompt: Prompt text for the model.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0.0 = greedy).
            top_p: Top-p sampling parameter.

        Returns:
            Raw model output string (JSON with layout elements).
        """
        # Build messages in Qwen2VL chat format
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                        "min_pixels": self.min_pixels,
                        "max_pixels": self.max_pixels,
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        # Apply chat template with thinking disabled
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # Process vision info using qwen_vl_utils
        image_inputs, _ = process_vision_info(messages, image_patch_size=16)

        # Tokenise text + images
        inputs = self.processor(
            text=text,
            images=image_inputs,
            do_resize=False,
            padding=True,
            return_tensors="pt",
        )
        # Remove token_type_ids if present (not needed for generation)
        inputs.pop("token_type_ids", None)

        # Move tensors to model device
        inputs = {
            k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        n_input_tokens = inputs["input_ids"].shape[1]
        logger.debug(
            f"Input tokens: {n_input_tokens}, max_new_tokens: {max_new_tokens}"
        )

        # Generate
        do_sample = temperature > 0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if not do_sample:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        # Strip input tokens, keeping only newly generated response
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs["input_ids"], output_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        logger.debug(f"Generated {generated_ids_trimmed[0].shape[0]} tokens")
        return output_text


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def parse_layout_json(json_str: str, width: int, height: int) -> OCRPage:
    """Parse the post-processed layout JSON into an OCRPage.

    The input should already have pixel-coordinate bboxes (after
    restore_abs_bbox_coordinates has been applied).

    Args:
        json_str: Post-processed JSON string with layout elements.
        width: Original image width in pixels.
        height: Original image height in pixels.

    Returns:
        OCRPage with regions and full_text populated.
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed: {e}")
        # Fall back to using the raw text as full_text
        return OCRPage(full_text=json_str, regions=[], width=width, height=height)

    # Handle list or dict wrapper
    items: list[dict[str, Any]] = []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        # Try common key names for the elements list
        for key in ("elements", "layouts", "layout", "results", "data"):
            if key in data and isinstance(data[key], list):
                items = data[key]
                break
        if not items:
            # Maybe the dict itself has bbox/category/text — wrap it
            if "category" in data or "bbox" in data:
                items = [data]

    regions: list[OCRRegion] = []
    full_text_parts: list[str] = []

    for order_idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        region = _parse_layout_item(item, order_idx, width, height)
        if region is None:
            continue

        regions.append(region)

        # Accumulate readable text (skip figures and page furniture)
        if region.text and region.category not in (
            "picture",
            "page-header",
            "page-footer",
        ):
            full_text_parts.append(region.text)

    full_text = "\n\n".join(full_text_parts)

    return OCRPage(
        full_text=full_text,
        regions=regions,
        width=width,
        height=height,
    )


def _parse_layout_item(
    item: dict[str, Any],
    order_idx: int,
    width: int,
    height: int,
) -> OCRRegion | None:
    """Parse a single layout element dict into an OCRRegion.

    Args:
        item: Dictionary with bbox, category, and text fields.
        order_idx: Reading order index.
        width: Image width (for bbox clamping).
        height: Image height (for bbox clamping).

    Returns:
        OCRRegion or None if the item is invalid.
    """
    # Parse category
    raw_category = str(item.get("category", "text")).strip().lower()
    category = CATEGORY_MAP.get(raw_category, raw_category)

    # Parse text
    text = item.get("text", "")
    if not isinstance(text, str):
        text = str(text) if text else ""

    # Parse bbox — at this point coordinates should already be in pixels
    bbox = None
    bbox_raw = item.get("bbox")
    if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) >= 4:
        x1, y1, x2, y2 = (
            float(bbox_raw[0]),
            float(bbox_raw[1]),
            float(bbox_raw[2]),
            float(bbox_raw[3]),
        )

        # Clamp to image bounds
        x1 = max(0.0, min(x1, float(width)))
        y1 = max(0.0, min(y1, float(height)))
        x2 = max(0.0, min(x2, float(width)))
        y2 = max(0.0, min(y2, float(height)))

        # Ensure proper ordering
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        # Only create BBox if it has non-zero area
        if x2 > x1 and y2 > y1:
            bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)

    # Determine text format based on category
    text_format = TEXT_FORMAT_MAP.get(category, "markdown")

    return OCRRegion(
        text=text,
        category=category,
        bbox=bbox,
        order=order_idx,
        text_format=text_format,
        confidence=-1.0,
    )


# ---------------------------------------------------------------------------
# Worker entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry-point invoked by the benchmark runner subprocess protocol."""
    parser = argparse.ArgumentParser(description="Infinity-Parser2 worker")
    parser.add_argument("--task", required=True, help="Path to input task JSON")
    parser.add_argument("--output", required=True, help="Path for output JSON")
    args = parser.parse_args()

    # Load task
    with open(args.task, "r", encoding="utf-8") as f:
        task = WorkerTask.from_dict(json.load(f))

    # Resolve device
    device = task.device if task.device else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        logger.warning("CUDA not available, falling back to CPU")

    # Get parameters from task (allow overrides)
    model_id = task.params.get("model_id", MODEL_ID)
    torch_dtype = task.params.get("torch_dtype", "float16")
    min_pixels = int(task.params.get("min_pixels", DEFAULT_MIN_PIXELS))
    max_pixels = int(task.params.get("max_pixels", DEFAULT_MAX_PIXELS))
    max_new_tokens = int(task.params.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
    temperature = float(task.params.get("temperature", 0.0))
    top_p = float(task.params.get("top_p", 1.0))

    # ── Memory tracking: before load ──
    ram_before = get_ram_mb()
    vram_before = get_vram_usage_mb()
    reset_peak_vram()

    # ── Load model ──
    t0 = time.perf_counter()
    model = InfinityParser2Model(
        model_id=model_id,
        device=device,
        torch_dtype=torch_dtype,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    model.load()
    load_time = time.perf_counter() - t0

    ram_after = get_ram_mb()
    vram_after = get_vram_usage_mb()

    logger.info(f"Model loaded in {load_time:.2f}s")
    logger.info(f"RAM: {ram_before:.0f} → {ram_after:.0f} MB")
    if vram_after is not None:
        logger.info(f"VRAM: {vram_before or 0:.0f} → {vram_after:.0f} MB")

    # ── Process pages ──
    pages: list[dict] = []
    for img_path in task.image_paths:
        logger.info(f"Processing: {img_path}")

        try:
            image = Image.open(img_path).convert("RGB")
            width, height = image.size

            t1 = time.perf_counter()
            raw_output = model.predict(
                image,
                prompt=PROMPT_DOC2JSON,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            pred_time = time.perf_counter() - t1

            logger.debug(f"Raw output ({len(raw_output)} chars): {raw_output[:500]}...")

            # Post-process: extract JSON, truncate if needed, restore bbox coords
            processed_json = postprocess_doc2json_result(raw_output, image)

            # Parse into OCRPage
            ocr_page = parse_layout_json(processed_json, width, height)
            ocr_page.page_number = len(pages) + 1

            pages.append(
                WorkerPageResult(
                    image_path=img_path,
                    prediction_time_seconds=pred_time,
                    ram_after_mb=get_ram_mb(),
                    result=ocr_page,
                ).to_dict()
            )

            n_regions = len(ocr_page.regions) if ocr_page.regions else 0
            logger.info(
                f"  → {n_regions} regions, "
                f"{len(ocr_page.full_text)} chars, "
                f"{pred_time:.2f}s"
            )

        except Exception as e:
            logger.exception(f"Failed to process {img_path}")
            pages.append(
                WorkerPageResult(
                    image_path=img_path,
                    error=str(e),
                    result=OCRPage(regions=[]),
                ).to_dict()
            )

    # ── Build response ──
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
