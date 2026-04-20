"""
Subprocess worker for Nanonets-OCR2 models (3B and 1.5B variants).

Loads the model from HuggingFace and runs inference for document OCR.
The specific model is selected via task.params["model_id"].

Protocol:
    python -m model_nanonets_ocr.worker --task task.json --output result.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
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

from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from ocr_core.types import OCRPage, OCRRegion, WorkerPageResult, WorkerTask
from ocr_core.utils import get_peak_vram_mb, get_vram_usage_mb, reset_peak_vram

DEFAULT_MODEL_ID = "nanonets/Nanonets-OCR2-3B"

DEFAULT_PROMPT = (
    "Extract the text from the above document as if you were reading it naturally. "
    "Return the tables in html format. Return the equations in LaTeX representation. "
    "If there is an image in the document and image caption is not present, add a "
    "small description of the image inside the <img></img> tag; otherwise, add the "
    "image caption inside <img></img>. Watermarks should be wrapped in brackets. "
    "Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. "  # noqa: E501
    "Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. "
    "Prefer using ☐ and ☑ for check boxes."
)


def get_ram_mb() -> float:
    """Get current RAM usage in MB."""
    return psutil.Process().memory_info().rss / (1024**2)


MIN_PIXELS = 3136  # 56×56 — Qwen2-VL minimum
MAX_PIXELS = 1003520  # ~1M pixels → max ~3600 vision tokens (fast prefill)
IMAGE_FACTOR = 28  # Qwen2-VL patch alignment factor


class NanonetsOCRModel:
    """Nanonets-OCR2 model wrapper (supports both 3B and 1.5B variants)."""

    def __init__(self, model_id: str, device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None
        self.tokenizer = None

    def load(self) -> None:
        """Load model, processor, and tokenizer from HuggingFace."""
        logger.info(f"Loading {self.model_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            cache_dir=CACHE_DIR,
        )

        # Constrain image size to avoid vision encoder hanging on large images
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            cache_dir=CACHE_DIR,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
        )

        if self.device == "cuda":
            dtype = torch.bfloat16
            try:
                import flash_attn  # noqa: F401

                attn_impl = "flash_attention_2"
            except ImportError:
                attn_impl = "sdpa"
                logger.warning("flash_attn not available, using SDPA")
        else:
            dtype = torch.float32
            attn_impl = "eager"

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            attn_implementation=attn_impl,
            cache_dir=CACHE_DIR,
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()
        logger.info(f"{self.model_id} loaded successfully")

    def predict(
        self,
        image_path: str,
        image: Image.Image,
        prompt: str = DEFAULT_PROMPT,
        max_new_tokens: int = 15000,
    ) -> str:
        """Run inference on a single image."""
        # Log image size for debugging
        logger.info(
            f"Image size: {image.size[0]}x{image.size[1]} "
            f"({image.size[0] * image.size[1]:,} pixels)"
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Log token counts for debugging
        n_input_tokens = inputs.input_ids.shape[1]
        logger.info(f"Input tokens: {n_input_tokens}, max_new_tokens: {max_new_tokens}")

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        logger.info(f"Generated {len(generated_ids[0])} tokens")
        return output_text


def parse_nanonets_output(text: str, width: int, height: int) -> OCRPage:
    """Parse Nanonets OCR output into an OCRPage.

    The model outputs text/markdown with:
    - HTML tables
    - LaTeX equations
    - <img>...</img> tags for images
    - <watermark>...</watermark> for watermarks
    - <page_number>...</page_number> for page numbers

    Args:
        text: Raw model output text.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Parsed OCRPage with regions extracted.
    """
    regions: list[OCRRegion] = []
    order_idx = 0

    # Extract tables (HTML)
    table_pattern = re.compile(r"(<table[\s\S]*?</table>)", re.IGNORECASE)
    for match in table_pattern.finditer(text):
        regions.append(
            OCRRegion(
                text=match.group(1),
                category="table",
                text_format="html",
                order=order_idx,
                confidence=-1.0,
            )
        )
        order_idx += 1

    # Extract LaTeX formulas ($$...$$, \[...\], \(...\))
    formula_patterns = [
        re.compile(r"\$\$(.+?)\$\$", re.DOTALL),
        re.compile(r"\\\[(.+?)\\\]", re.DOTALL),
        re.compile(r"\\\((.+?)\\\)", re.DOTALL),
    ]
    for pattern in formula_patterns:
        for match in pattern.finditer(text):
            regions.append(
                OCRRegion(
                    text=match.group(1).strip(),
                    category="formula",
                    text_format="latex",
                    order=order_idx,
                    confidence=-1.0,
                )
            )
            order_idx += 1

    # Extract image descriptions
    img_pattern = re.compile(r"<img>(.*?)</img>", re.DOTALL | re.IGNORECASE)
    for match in img_pattern.finditer(text):
        regions.append(
            OCRRegion(
                text=match.group(1).strip(),
                category="picture",
                text_format="plain",
                order=order_idx,
                confidence=-1.0,
            )
        )
        order_idx += 1

    # Extract watermarks
    watermark_pattern = re.compile(
        r"<watermark>(.*?)</watermark>", re.DOTALL | re.IGNORECASE
    )
    for match in watermark_pattern.finditer(text):
        regions.append(
            OCRRegion(
                text=match.group(1).strip(),
                category="page-header",
                text_format="plain",
                order=order_idx,
                confidence=-1.0,
            )
        )
        order_idx += 1

    # Extract page numbers
    page_num_pattern = re.compile(
        r"<page_number>(.*?)</page_number>", re.DOTALL | re.IGNORECASE
    )
    for match in page_num_pattern.finditer(text):
        regions.append(
            OCRRegion(
                text=match.group(1).strip(),
                category="page-footer",
                text_format="plain",
                order=order_idx,
                confidence=-1.0,
            )
        )
        order_idx += 1

    # Build full_text: strip special tags but keep the readable content
    full_text = text
    full_text = re.sub(
        r"<img>(.*?)</img>", r"\1", full_text, flags=re.DOTALL | re.IGNORECASE
    )
    full_text = re.sub(
        r"<watermark>(.*?)</watermark>",
        r"\1",
        full_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    full_text = re.sub(
        r"<page_number>(.*?)</page_number>",
        r"\1",
        full_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    full_text = full_text.strip()

    return OCRPage(
        full_text=full_text,
        regions=regions,
        width=width,
        height=height,
    )


def main():
    """Main entry point for worker subprocess."""
    parser = argparse.ArgumentParser(description="Nanonets-OCR2 worker")
    parser.add_argument("--task", required=True, help="Path to task JSON")
    parser.add_argument("--output", required=True, help="Path for output JSON")
    args = parser.parse_args()

    with open(args.task, "r", encoding="utf-8") as f:
        task_data = json.load(f)
    task = WorkerTask.from_dict(task_data)

    # Resolve device
    device = task.device if task.device else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        logger.warning("CUDA not available, falling back to CPU")

    # Model ID from params (allows 3B vs 1.5B selection)
    model_id = task.params.get("model_id", DEFAULT_MODEL_ID)

    # Memory tracking - before load
    ram_before = get_ram_mb()
    vram_before = get_vram_usage_mb()
    reset_peak_vram()

    # Load model
    t0 = time.perf_counter()
    model = NanonetsOCRModel(model_id=model_id, device=device)
    model.load()
    load_time = time.perf_counter() - t0

    ram_after = get_ram_mb()
    vram_after = get_vram_usage_mb()

    logger.info(f"Model loaded in {load_time:.2f}s")
    logger.info(f"RAM: {ram_before:.0f} → {ram_after:.0f} MB")
    if vram_after is not None:
        logger.info(f"VRAM: {vram_before or 0:.0f} → {vram_after:.0f} MB")

    # Get inference parameters from task params
    prompt = task.params.get("prompt", DEFAULT_PROMPT)
    max_new_tokens = task.params.get("max_new_tokens", 15000)

    # Process images
    pages: list[dict] = []
    for img_path in task.image_paths:
        logger.info(f"Processing: {img_path}")

        try:
            image = Image.open(img_path).convert("RGB")
            width, height = image.size

            t1 = time.perf_counter()
            output_text = model.predict(
                image_path=img_path,
                image=image,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
            pred_time = time.perf_counter() - t1

            logger.debug(
                f"Raw output ({len(output_text)} chars): {output_text[:500]}..."
            )

            ocr_page = parse_nanonets_output(output_text, width, height)
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
                f"{len(ocr_page.full_text)} chars, {pred_time:.2f}s"
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
