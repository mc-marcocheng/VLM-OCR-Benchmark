"""
Subprocess worker for LightOnOCR-2.

Loads a LightOnOCR-2 model variant and runs OCR inference.  Supports both
plain-text models (e.g. ``lightonai/LightOnOCR-2-1B``) and bbox-annotated
models (e.g. ``lightonai/LightOnOCR-2-1B-bbox``).

The model receives only an image (no text prompt) and returns markdown.
Bbox variants additionally annotate detected image regions with normalised
[0-1000] bounding boxes in the pattern ``![image](image_N.png)x1,y1,x2,y2``.

Protocol:
    python -m model_lighton_ocr2 --task task.json --output result.json
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

from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

from model_lighton_ocr2.utils import (
    clean_output_text,
    parse_bbox_to_regions,
    parse_plain_to_regions,
)
from ocr_core.types import BBox, OCRPage, OCRRegion, WorkerPageResult, WorkerTask
from ocr_core.utils import get_peak_vram_mb, get_vram_usage_mb, reset_peak_vram

DEFAULT_MODEL_ID = "lightonai/LightOnOCR-2-1B"
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 0.9


def get_ram_mb() -> float:
    """Get current process RAM usage in MB."""
    return psutil.Process().memory_info().rss / (1024**2)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class LightOnOCR2Model:
    """Wrapper around LightOnOCR-2 for document OCR inference.

    Handles both plain and bbox model variants via the ``has_bbox`` flag.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        has_bbox: bool = False,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype, torch.bfloat16)
        self.has_bbox = has_bbox
        self.model = None
        self.processor = None

    def load(self) -> None:
        """Load model weights and processor from HuggingFace."""
        logger.info(f"Loading LightOnOCR-2 from {self.model_id} (bbox={self.has_bbox})")

        # Determine attention implementation
        if self.device == "cuda":
            try:
                import flash_attn  # noqa: F401

                attn_impl = "flash_attention_2"
                logger.info("Using flash_attention_2 backend")
            except ImportError:
                attn_impl = "sdpa"
                logger.info("Using SDPA backend")
        else:
            attn_impl = "eager"
            logger.info("Using eager attention (CPU)")

        self.model = (
            LightOnOcrForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                attn_implementation=attn_impl,
                trust_remote_code=True,
                cache_dir=CACHE_DIR,
            )
            .to(self.device)
            .eval()
        )

        self.processor = LightOnOcrProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
        )

        logger.info("LightOnOCR-2 loaded successfully")

    def predict(
        self,
        image: Image.Image,
        *,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        """Run OCR inference on a single page image.

        The model receives **only an image** (no text prompt).  The chat
        template wraps it in the expected format automatically.

        Args:
            image: Document page as a PIL Image (RGB).
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 = greedy/deterministic).
            top_p: Top-p sampling parameter.

        Returns:
            Cleaned model output string (markdown, possibly with bbox annotations).
        """
        # Build chat: image only, no text prompt
        chat = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image},
                ],
            }
        ]

        # Tokenise via chat template (handles image preprocessing internally)
        inputs = self.processor.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Move tensors to device with correct dtype
        inputs = {
            k: (
                v.to(device=self.model.device, dtype=self.torch_dtype)
                if isinstance(v, torch.Tensor)
                and v.dtype in (torch.float32, torch.float16, torch.bfloat16)
                else v.to(self.model.device) if isinstance(v, torch.Tensor) else v
            )
            for k, v in inputs.items()
        }

        n_input_tokens = inputs["input_ids"].shape[1]
        logger.debug(
            f"Input tokens: {n_input_tokens}, max_new_tokens: {max_new_tokens}"
        )

        do_sample = temperature > 0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if do_sample else 0.0,
            "top_p": top_p,
            "top_k": 0,
            "do_sample": do_sample,
            "use_cache": True,
        }

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        # Trim input tokens from output
        generated_ids = output_ids[0][n_input_tokens:]
        output_text = self.processor.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # Safety cleanup for any remaining chat template artifacts
        output_text = clean_output_text(output_text)

        logger.debug(f"Generated {len(generated_ids)} tokens")
        return output_text


# ---------------------------------------------------------------------------
# Output → OCRPage
# ---------------------------------------------------------------------------


def build_ocr_page(
    raw_text: str,
    image_width: int,
    image_height: int,
    has_bbox: bool,
) -> OCRPage:
    """Convert model output text to an OCRPage.

    Args:
        raw_text: Cleaned model output (markdown ± bbox annotations).
        image_width: Original image width in pixels.
        image_height: Original image height in pixels.
        has_bbox: Whether the model outputs bounding box annotations.

    Returns:
        Populated OCRPage.
    """
    if has_bbox:
        full_text, region_dicts = parse_bbox_to_regions(
            raw_text, image_width, image_height
        )
    else:
        full_text, region_dicts = parse_plain_to_regions(raw_text)

    regions: list[OCRRegion] = []
    for rd in region_dicts:
        bbox = None
        if rd["bbox"] is not None:
            x1, y1, x2, y2 = rd["bbox"]
            bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)

        regions.append(
            OCRRegion(
                text=rd["text"],
                category=rd["category"],
                bbox=bbox,
                order=rd["order"],
                text_format=rd["text_format"],
                confidence=-1.0,
            )
        )

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
    parser = argparse.ArgumentParser(description="LightOnOCR-2 worker")
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
    model_id = task.params.get("model_id", DEFAULT_MODEL_ID)
    torch_dtype = task.params.get("torch_dtype", "bfloat16")
    has_bbox = task.params.get("has_bbox", False)
    max_new_tokens = int(task.params.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
    temperature = float(task.params.get("temperature", DEFAULT_TEMPERATURE))
    top_p = float(task.params.get("top_p", DEFAULT_TOP_P))

    # ── Memory tracking: before load ──
    ram_before = get_ram_mb()
    vram_before = get_vram_usage_mb()
    reset_peak_vram()

    # ── Load model ──
    t0 = time.perf_counter()
    model = LightOnOCR2Model(
        model_id=model_id,
        device=device,
        torch_dtype=torch_dtype,
        has_bbox=has_bbox,
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
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            pred_time = time.perf_counter() - t1

            logger.debug(f"Raw output ({len(raw_output)} chars): {raw_output[:500]}...")

            # Parse into OCRPage
            ocr_page = build_ocr_page(raw_output, width, height, has_bbox)
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
