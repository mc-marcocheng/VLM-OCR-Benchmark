"""
Subprocess worker for Chandra OCR.

Loads the datalab-to/chandra-ocr-2 model (a fine-tuned Qwen2.5-VL variant)
and runs layout-aware OCR inference. The model outputs HTML with layout blocks
containing bounding boxes (normalised to [0-1000]), labels, and formatted text
content (HTML with embedded LaTeX in <math> tags, tables as <table>, etc.).

Protocol:
    python -m model_chandra_ocr --task task.json --output result.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

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

from transformers import AutoModelForImageTextToText, AutoProcessor

from model_chandra_ocr.utils import OCR_LAYOUT_PROMPT, parse_html_layout, scale_to_fit
from model_chandra_ocr.utils.postprocess import (
    extract_latex_from_math_tags,
    extract_table_html,
    extract_text_from_html,
)
from ocr_core.types import BBox, OCRPage, OCRRegion, WorkerPageResult, WorkerTask
from ocr_core.utils import get_peak_vram_mb, get_vram_usage_mb, reset_peak_vram

MODEL_ID = "datalab-to/chandra-ocr-2"
DEFAULT_MAX_OUTPUT_TOKENS = 12384

# Category mapping from Chandra labels to ocr-core standard labels
CATEGORY_MAP: dict[str, str] = {
    "caption": "caption",
    "footnote": "footnote",
    "equation-block": "formula",
    "list-group": "list-item",
    "page-header": "page-header",
    "page-footer": "page-footer",
    "image": "picture",
    "section-header": "section-header",
    "table": "table",
    "text": "text",
    "complex-block": "text",
    "code-block": "code",
    "form": "text",
    "table-of-contents": "text",
    "figure": "picture",
    "chemical-block": "formula",
    "diagram": "picture",
    "bibliography": "text",
}

# Map categories to text format for OCRRegion
TEXT_FORMAT_MAP: dict[str, str] = {
    "table": "html",
    "formula": "latex",
    "code": "plain",
    "picture": "plain",
}


def get_ram_mb() -> float:
    """Get current process RAM usage in MB."""
    return psutil.Process().memory_info().rss / (1024**2)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class ChandraOCRModel:
    """Wrapper around the Chandra OCR VLM for layout-aware document OCR."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype, torch.bfloat16)
        self.model = None
        self.processor = None

    def load(self) -> None:
        """Load model weights and processor from HuggingFace."""
        logger.info(f"Loading Chandra OCR from {self.model_id}")

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

        device_map = "auto" if self.device == "cuda" else None

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            device_map=device_map,
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
        # Left-padding is required for batch generation
        self.processor.tokenizer.padding_side = "left"

        # Attach processor to model for convenience
        self.model.processor = self.processor

        logger.info("Chandra OCR loaded successfully")

    def predict(
        self,
        image: Image.Image,
        prompt: str = OCR_LAYOUT_PROMPT,
        *,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> str:
        """Run layout-aware OCR inference on a single page.

        Args:
            image: Document page as a PIL Image (RGB).
            prompt: Prompt text for the model.
            max_output_tokens: Maximum number of tokens to generate.

        Returns:
            Raw model output string (HTML with layout div blocks).
        """
        # Resize to fit model constraints
        image = scale_to_fit(image)

        # Build conversation in multimodal chat format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # apply_chat_template with tokenize=True handles both text and images
        inputs = self.processor.apply_chat_template(
            [conversation],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(self.model.device)

        n_input_tokens = inputs["input_ids"].shape[1]
        logger.debug(
            f"Input tokens: {n_input_tokens}, max_output_tokens: {max_output_tokens}"
        )

        # Build EOS token list: include both <|endoftext|> and <|im_end|>
        eos_token_id = self.model.generation_config.eos_token_id
        im_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        elif eos_token_id is None:
            eos_token_id = []
        if im_end_id is not None and im_end_id not in eos_token_id:
            eos_token_id.append(im_end_id)

        # Generate
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_output_tokens,
                eos_token_id=eos_token_id,
            )

        # Strip input tokens, keeping only generated response
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
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


def build_ocr_page(
    raw_html: str,
    image_width: int,
    image_height: int,
) -> OCRPage:
    """Parse raw HTML model output into an OCRPage.

    Args:
        raw_html: Raw HTML output from the model.
        image_width: Original image width in pixels.
        image_height: Original image height in pixels.

    Returns:
        OCRPage with regions and full_text populated.
    """
    layout_blocks = parse_html_layout(raw_html, image_width, image_height)

    regions: list[OCRRegion] = []
    full_text_parts: list[str] = []

    for order_idx, block in enumerate(layout_blocks):
        # Map label to standard category
        raw_label = block.label.lower().strip()
        category = CATEGORY_MAP.get(raw_label, "text")

        # Determine text format and extract text accordingly
        text_format = TEXT_FORMAT_MAP.get(category, "html")

        if category == "formula":
            text = extract_latex_from_math_tags(block.content)
        elif category == "table":
            text = extract_table_html(block.content)
        elif category == "picture":
            # For images/figures, extract alt text or description
            text = extract_text_from_html(block.content)
            text_format = "plain"
        else:
            # Keep as HTML for rich content (preserves formatting)
            text = block.content.strip()
            text_format = "html"

        # Build BBox (block.bbox is [x1, y1, x2, y2] in pixel coords)
        x1, y1, x2, y2 = block.bbox
        bbox = None
        if x2 > x1 and y2 > y1:
            bbox = BBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))

        region = OCRRegion(
            text=text,
            category=category,
            bbox=bbox,
            order=order_idx,
            text_format=text_format,
            confidence=-1.0,
        )
        regions.append(region)

        # Accumulate readable text (skip images and page furniture)
        if category not in ("picture", "page-header", "page-footer"):
            plain_text = extract_text_from_html(block.content)
            if plain_text:
                full_text_parts.append(plain_text)

    full_text = "\n\n".join(full_text_parts)

    return OCRPage(
        full_text=full_text,
        regions=regions,
        width=image_width,
        height=image_height,
    )


# ---------------------------------------------------------------------------
# Worker entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry-point invoked by the benchmark runner subprocess protocol."""
    parser = argparse.ArgumentParser(description="Chandra OCR worker")
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

    # Get parameters from task
    model_id = task.params.get("model_id", MODEL_ID)
    torch_dtype = task.params.get("torch_dtype", "bfloat16")
    max_output_tokens = int(
        task.params.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS)
    )

    # ── Memory tracking: before load ──
    ram_before = get_ram_mb()
    vram_before = get_vram_usage_mb()
    reset_peak_vram()

    # ── Load model ──
    t0 = time.perf_counter()
    model = ChandraOCRModel(
        model_id=model_id,
        device=device,
        torch_dtype=torch_dtype,
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
                prompt=OCR_LAYOUT_PROMPT,
                max_output_tokens=max_output_tokens,
            )
            pred_time = time.perf_counter() - t1

            logger.debug(f"Raw output ({len(raw_output)} chars): {raw_output[:500]}...")

            # Parse HTML output into OCRPage
            ocr_page = build_ocr_page(raw_output, width, height)
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
