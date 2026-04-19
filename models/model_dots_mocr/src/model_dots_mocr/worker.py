"""
Subprocess worker for Dots.MOCR.

Loads the model directly from HuggingFace and runs inference.

Protocol:
    python -m model_dots_mocr.worker --task task.json --output result.json
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

# Set environment variables BEFORE importing transformers to strictly enforce the cache location
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_HUB_CACHE"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
# ─────────────────────────────────────────────────────────────

from model_dots_mocr.utils import (
    MAX_PIXELS,
    MIN_PIXELS,
    dict_promptmode_to_prompt,
    smart_resize,
)
from ocr_core.types import BBox, OCRPage, OCRRegion, WorkerPageResult, WorkerTask
from ocr_core.utils import get_peak_vram_mb, get_vram_usage_mb, reset_peak_vram
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForCausalLM, AutoProcessor

MODEL_ID = "rednote-hilab/dots.mocr"

# Category mapping from Dots.MOCR labels to ocr-core standard labels
CATEGORY_MAP = {
    "Text": "text",
    "Title": "title",
    "Section-header": "section-header",
    "Caption": "caption",
    "Footnote": "footnote",
    "Page-header": "page-header",
    "Page-footer": "page-footer",
    "List-item": "list-item",
    "Table": "table",
    "Formula": "formula",
    "Picture": "picture",
}


def get_ram_mb() -> float:
    """Get current RAM usage in MB."""
    return psutil.Process().memory_info().rss / (1024**2)


class DotsMOCRModel:
    """Dots.MOCR model wrapper for direct HuggingFace inference."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.processor = None

    def load(self) -> None:
        """Load model and processor from HuggingFace."""
        logger.info(f"Loading Dots.MOCR from {MODEL_ID}")

        # Load processor with custom code
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
            cache_dir=CACHE_DIR,
        )

        # Determine dtype and attention implementation based on device
        if self.device == "cuda":
            dtype = torch.bfloat16
            # Check if flash attention is available
            try:
                import flash_attn  # noqa: F401

                attn_impl = "flash_attention_2"
            except ImportError:
                attn_impl = "sdpa"
                logger.warning("flash_attn not available, using SDPA")
        else:
            dtype = torch.float32
            attn_impl = "eager"

        # Use AutoModelForCausalLM with trust_remote_code to load the custom dots_ocr model
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            cache_dir=CACHE_DIR,
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()
        logger.info("Dots.MOCR loaded successfully")

    def predict(
        self,
        image: Image.Image,
        prompt_mode: str = "prompt_layout_all_en",
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_new_tokens: int = 32768,
    ) -> str:
        """Run inference on a single image.

        Args:
            image: PIL Image to process.
            prompt_mode: Which prompt to use from dict_promptmode_to_prompt.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Model output text.
        """
        prompt = dict_promptmode_to_prompt.get(
            prompt_mode, dict_promptmode_to_prompt["prompt_layout_all_en"]
        )

        # Resize image to fit model constraints
        image = smart_resize(image, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)

        # Build messages in Qwen2VL format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process vision info
        image_inputs, video_inputs = process_vision_info(messages)

        # Prepare inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Generate
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p if temperature > 0 else None,
            )

        # Decode - trim input tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return output_text


def extract_json_from_response(text: str) -> str:
    """Extract JSON content from model response."""
    text = text.strip()

    # Remove markdown code blocks
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Try to find JSON array or object
    start_bracket = -1
    for i, c in enumerate(text):
        if c in "[{":
            start_bracket = i
            break

    if start_bracket >= 0:
        # Find matching closing bracket
        open_char = text[start_bracket]
        close_char = "]" if open_char == "[" else "}"
        depth = 0
        end_bracket = -1
        for i in range(start_bracket, len(text)):
            if text[i] == open_char:
                depth += 1
            elif text[i] == close_char:
                depth -= 1
                if depth == 0:
                    end_bracket = i
                    break

        if end_bracket > start_bracket:
            text = text[start_bracket : end_bracket + 1]

    return text


def parse_layout_json(json_str: str, width: int, height: int) -> OCRPage:
    """Parse the layout JSON output into OCRPage."""
    json_str = extract_json_from_response(json_str)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed: {e}")
        return OCRPage(full_text=json_str, width=width, height=height)

    regions: list[OCRRegion] = []
    full_texts: list[str] = []

    # Handle different output formats
    items: list[dict[str, Any]] = []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        if "layout" in data:
            items = data["layout"]
        else:
            items = [data]

    for idx, item in enumerate(items):
        region = parse_layout_item(item, idx, width, height)
        if region:
            regions.append(region)
            if region.text and region.category not in (
                "picture",
                "page-header",
                "page-footer",
            ):
                full_texts.append(region.text)

    full_text = "\n\n".join(full_texts)

    return OCRPage(
        full_text=full_text,
        regions=regions,
        width=width,
        height=height,
    )


def parse_layout_item(
    item: dict[str, Any], idx: int, width: int, height: int
) -> OCRRegion | None:
    """Parse a single layout item into OCRRegion."""
    if not isinstance(item, dict):
        return None

    # Parse bbox
    bbox = None
    if "bbox" in item:
        b = item["bbox"]
        if isinstance(b, (list, tuple)) and len(b) >= 4:
            x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])

            # Check if coordinates are normalized (0-1 range)
            if all(0 <= v <= 1 for v in [x1, y1, x2, y2]):
                x1 *= width
                y1 *= height
                x2 *= width
                y2 *= height

            # Ensure proper ordering
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)

    # Parse category
    raw_category = item.get("category", item.get("type", "Text"))
    if isinstance(raw_category, str):
        category = CATEGORY_MAP.get(raw_category, raw_category.lower())
    else:
        category = "text"

    # Parse text
    text = item.get("text", "")
    if not isinstance(text, str):
        text = str(text) if text else ""

    # Determine text format based on category
    text_format = "plain"
    if category == "table":
        text_format = "html"
    elif category == "formula":
        text_format = "latex"
    elif text and any(marker in text for marker in ["**", "##", "- ", "* "]):
        text_format = "markdown"

    # Parse confidence if available
    confidence = item.get("confidence", item.get("score", -1.0))
    if not isinstance(confidence, (int, float)):
        confidence = -1.0

    return OCRRegion(
        text=text,
        category=category,
        bbox=bbox,
        order=idx,
        text_format=text_format,
        confidence=float(confidence),
    )


def main():
    """Main entry point for worker subprocess."""
    parser = argparse.ArgumentParser(description="Dots.MOCR worker")
    parser.add_argument("--task", required=True, help="Path to task JSON")
    parser.add_argument("--output", required=True, help="Path for output JSON")
    args = parser.parse_args()

    # Load task
    with open(args.task, "r", encoding="utf-8") as f:
        task_data = json.load(f)
    task = WorkerTask.from_dict(task_data)

    # Resolve device
    device = task.device if task.device else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        logger.warning("CUDA not available, falling back to CPU")

    # Memory tracking - before load
    ram_before = get_ram_mb()
    vram_before = get_vram_usage_mb()
    reset_peak_vram()

    # Load model
    t0 = time.perf_counter()
    model = DotsMOCRModel(device=device)
    model.load()
    load_time = time.perf_counter() - t0

    ram_after = get_ram_mb()
    vram_after = get_vram_usage_mb()

    logger.info(f"Model loaded in {load_time:.2f}s")
    logger.info(f"RAM: {ram_before:.0f} → {ram_after:.0f} MB")
    if vram_after is not None:
        logger.info(f"VRAM: {vram_before or 0:.0f} → {vram_after:.0f} MB")

    # Get inference parameters from task params
    prompt_mode = task.params.get("prompt_mode", "prompt_layout_all_en")
    temperature = task.params.get("temperature", 0.1)
    top_p = task.params.get("top_p", 0.9)
    max_new_tokens = task.params.get("max_new_tokens", 32768)

    # Process images
    pages: list[dict] = []
    for img_path in task.image_paths:
        logger.info(f"Processing: {img_path}")

        try:
            image = Image.open(img_path).convert("RGB")
            width, height = image.size

            t1 = time.perf_counter()
            output_text = model.predict(
                image,
                prompt_mode=prompt_mode,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
            pred_time = time.perf_counter() - t1

            logger.debug(
                f"Raw output ({len(output_text)} chars): {output_text[:500]}..."
            )

            # Parse output into OCRPage
            ocr_page = parse_layout_json(output_text, width, height)
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

        except Exception as e:
            logger.exception(f"Failed to process {img_path}")
            pages.append(
                WorkerPageResult(
                    image_path=img_path,
                    error=str(e),
                    result=OCRPage(),
                ).to_dict()
            )

    # Build response
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
