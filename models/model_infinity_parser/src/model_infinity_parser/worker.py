"""
Subprocess worker for Infinity-Parser.

Loads the Infinity-Parser-7B model (a Qwen2.5-VL variant fine-tuned with
layout-aware reinforcement learning) and runs document-to-Markdown OCR
inference.  Unlike Infinity-Parser2, this model outputs **Markdown directly**
and does NOT produce layout bounding boxes or structured JSON.

Protocol:
    python -m model_infinity_parser.worker --task task.json --output result.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Any

# ── Configure HuggingFace Cache ──────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))
CACHE_DIR = os.path.join(_REPO_ROOT, "models", "huggingface_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_HUB_CACHE"] = CACHE_DIR
# ─────────────────────────────────────────────────────────────

import psutil
import torch
from loguru import logger
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from model_infinity_parser.utils.prompt import PROMPT
from ocr_core.types import OCRPage, WorkerPageResult, WorkerTask
from ocr_core.utils import get_peak_vram_mb, get_vram_usage_mb, reset_peak_vram

MODEL_ID = "infly/Infinity-Parser-7B"
DEFAULT_MIN_PIXELS = 200704  # 256 * 28 * 28
DEFAULT_MAX_PIXELS = 1806336  # 2304 * 28 * 28
DEFAULT_MAX_NEW_TOKENS = 8192


def get_ram_mb() -> float:
    """Get current process RAM usage in MB."""
    return psutil.Process().memory_info().rss / (1024**2)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


def extract_markdown_content(text: str) -> str:
    """Strip ```markdown fences from model output if present.

    Args:
        text: Raw model output text.

    Returns:
        Cleaned markdown string.
    """
    match = re.search(r"```markdown\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class InfinityParserModel:
    """Wrapper around Infinity-Parser-7B for document-to-Markdown OCR."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        min_pixels: int = DEFAULT_MIN_PIXELS,
        max_pixels: int = DEFAULT_MAX_PIXELS,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype, torch.bfloat16)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.model = None
        self.processor = None

    def load(self) -> None:
        """Load model weights and processor from HuggingFace."""
        logger.info(f"Loading Infinity-Parser from {self.model_id}")

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

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            cache_dir=CACHE_DIR,
        )

        logger.info("Infinity-Parser loaded successfully")

    def predict(
        self,
        image: Image.Image,
        prompt: str = PROMPT,
        *,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = 0.0,
        top_p: float = 0.95,
    ) -> str:
        """Run document-to-Markdown OCR inference on a single page.

        Args:
            image: Document page as a PIL Image (RGB).
            prompt: Prompt text for the model.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0.0 = greedy).
            top_p: Top-p sampling parameter.

        Returns:
            Raw model output string (Markdown text).
        """
        # Build messages in Qwen2.5-VL chat format
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

        # Apply chat template (v4 API — no enable_thinking parameter)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process vision info using qwen_vl_utils (older API — no image_patch_size)
        image_inputs, video_inputs = process_vision_info(messages)

        # Tokenise text + images
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Move tensors to model device
        inputs = inputs.to(self.model.device)

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


def parse_output(raw_text: str, width: int, height: int) -> OCRPage:
    """Parse raw model output into an OCRPage.

    Infinity-Parser outputs Markdown directly (no layout/bbox info),
    so we just extract the text and return an OCRPage with empty regions.

    Args:
        raw_text: Raw model output string.
        width: Original image width in pixels.
        height: Original image height in pixels.

    Returns:
        OCRPage with full_text populated and empty regions list.
    """
    text = extract_markdown_content(raw_text)
    return OCRPage(full_text=text, regions=[], width=width, height=height)


# ---------------------------------------------------------------------------
# Worker entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry-point invoked by the benchmark runner subprocess protocol."""
    parser = argparse.ArgumentParser(description="Infinity-Parser worker")
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
    torch_dtype = task.params.get("torch_dtype", "bfloat16")
    min_pixels = int(task.params.get("min_pixels", DEFAULT_MIN_PIXELS))
    max_pixels = int(task.params.get("max_pixels", DEFAULT_MAX_PIXELS))
    max_new_tokens = int(task.params.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
    temperature = float(task.params.get("temperature", 0.0))
    top_p = float(task.params.get("top_p", 0.95))

    # ── Memory tracking: before load ──
    ram_before = get_ram_mb()
    vram_before = get_vram_usage_mb()
    reset_peak_vram()

    # ── Load model ──
    t0 = time.perf_counter()
    model = InfinityParserModel(
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
                prompt=PROMPT,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            pred_time = time.perf_counter() - t1

            logger.debug(f"Raw output ({len(raw_output)} chars): {raw_output[:500]}...")

            # Parse into OCRPage (just markdown extraction, no bbox processing)
            ocr_page = parse_output(raw_output, width, height)
            ocr_page.page_number = len(pages) + 1

            pages.append(
                WorkerPageResult(
                    image_path=img_path,
                    prediction_time_seconds=pred_time,
                    ram_after_mb=get_ram_mb(),
                    result=ocr_page,
                ).to_dict()
            )

            logger.info(f"  → {len(ocr_page.full_text)} chars, {pred_time:.2f}s")

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
