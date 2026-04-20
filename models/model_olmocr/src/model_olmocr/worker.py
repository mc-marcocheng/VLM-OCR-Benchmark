"""
Subprocess worker for olmOCR-2.

Loads the olmOCR-2-7B-1025-FP8 model (a fine-tuned Qwen2.5-VL-7B) and runs
document OCR inference. Uses the official olmocr prompt builder for the
no-anchoring v4 YAML prompt format.

The model outputs YAML front-matter (metadata such as language, rotation,
whether the page is a table/diagram) followed by clean Markdown text of the
document content.

Protocol:
    python -m model_olmocr.worker --task task.json --output result.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from io import BytesIO

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

# olmOCR prompt builder — the canonical way to build the input prompt
from olmocr.prompts import build_no_anchoring_v4_yaml_prompt
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from ocr_core.types import OCRPage, OCRRegion, WorkerPageResult, WorkerTask
from ocr_core.utils import get_peak_vram_mb, get_vram_usage_mb, reset_peak_vram

DEFAULT_MODEL_ID = "allenai/olmOCR-2-7B-1025-FP8"
DEFAULT_PROCESSOR_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
TARGET_LONGEST_IMAGE_DIM = 1288


def get_ram_mb() -> float:
    """Get current RAM usage in MB."""
    return psutil.Process().memory_info().rss / (1024**2)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def resize_to_target(
    image: Image.Image,
    target_longest_dim: int = TARGET_LONGEST_IMAGE_DIM,
) -> Image.Image:
    """Resize *image* so its longest side equals *target_longest_dim*.

    If the image is already smaller than the target, it is returned unchanged.
    """
    w, h = image.size
    longest = max(w, h)
    if longest <= target_longest_dim:
        return image
    scale = target_longest_dim / longest
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    return image.resize((new_w, new_h), Image.LANCZOS)


def image_to_base64png(image: Image.Image) -> str:
    """Encode a PIL Image as a base-64 PNG string."""
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class OlmOCRModel:
    """Wrapper around the olmOCR-2 VLM for single-page document OCR."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        processor_id: str = DEFAULT_PROCESSOR_ID,
        device: str = "cuda",
    ):
        self.model_id = model_id
        self.processor_id = processor_id
        self.device = device
        self.model = None
        self.processor = None

    def load(self) -> None:
        """Load model weights and processor from HuggingFace."""
        logger.info(f"Loading olmOCR-2 model from {self.model_id}")
        logger.info(f"Loading processor from {self.processor_id}")

        # Processor — always from the base Qwen2.5-VL-7B-Instruct
        self.processor = AutoProcessor.from_pretrained(
            self.processor_id,
            cache_dir=CACHE_DIR,
        )

        # Determine dtype / attention backend
        if self.device == "cuda":
            try:
                import flash_attn  # noqa: F401

                attn_impl = "flash_attention_2"
            except ImportError:
                attn_impl = "sdpa"
                logger.warning("flash_attn not available, falling back to SDPA")
        else:
            attn_impl = "eager"

        # The FP8 checkpoint is loaded with device_map="auto" which places
        # layers across available GPUs and handles the FP8 de-quantisation
        # transparently via the transformers compressed-tensors integration.
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map="auto" if self.device == "cuda" else None,
            attn_implementation=attn_impl,
            cache_dir=CACHE_DIR,
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()
        logger.info("olmOCR-2 model loaded successfully")

    def predict(
        self,
        image: Image.Image,
        *,
        target_longest_dim: int = TARGET_LONGEST_IMAGE_DIM,
        temperature: float = 0.1,
        max_new_tokens: int = 16384,
    ) -> str:
        """Run single-page OCR inference.

        Args:
            image: Document page as a PIL Image (RGB).
            target_longest_dim: Resize so the longest side equals this value.
            temperature: Sampling temperature (0.1 recommended by olmOCR).
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            Raw model output string (YAML front-matter + Markdown body).
        """
        # 1. Resize to the target dimension expected by the model
        image = resize_to_target(image, target_longest_dim)
        logger.debug(
            f"Image after resize: {image.size[0]}×{image.size[1]} "
            f"({image.size[0] * image.size[1]:,} px)"
        )

        # 2. Encode to base-64 PNG (olmOCR uses data-URI image format)
        image_b64 = image_to_base64png(image)

        # 3. Build the prompt using the official olmOCR prompt builder
        prompt_text = build_no_anchoring_v4_yaml_prompt()

        # 4. Assemble the chat messages (text-first, then image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                        },
                    },
                ],
            },
        ]

        # 5. Apply the chat template to get the tokeniser-ready string
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 6. Decode the base-64 back to a PIL image for the vision encoder
        #    (mirrors the official sample code exactly)
        main_image = Image.open(BytesIO(base64.b64decode(image_b64)))

        # 7. Tokenise text + image
        inputs = self.processor(
            text=[text],
            images=[main_image],
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        n_input_tokens = inputs["input_ids"].shape[1]
        logger.debug(
            f"Input tokens: {n_input_tokens}, max_new_tokens: {max_new_tokens}"
        )

        # 8. Generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                do_sample=temperature > 0,
            )

        # 9. Trim input tokens and decode
        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = output_ids[:, prompt_length:]
        output_text = self.processor.tokenizer.batch_decode(
            new_tokens, skip_special_tokens=True
        )[0]

        logger.debug(f"Generated {new_tokens.shape[1]} tokens")
        return output_text


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def parse_yaml_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split YAML front-matter from the Markdown body.

    Returns:
        A tuple of (metadata_dict, body_text). If no front-matter is found
        the metadata dict is empty and body_text is the original text.
    """
    text = text.strip()
    metadata: dict[str, str] = {}

    if not text.startswith("---"):
        return metadata, text

    # Find the closing '---' (skip the opening one)
    end_idx = text.find("---", 3)
    if end_idx == -1:
        return metadata, text

    yaml_block = text[3:end_idx].strip()
    body = text[end_idx + 3 :].strip()

    for line in yaml_block.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        metadata[key.strip()] = value.strip()

    return metadata, body


def parse_olmocr_output(raw_output: str, width: int, height: int) -> OCRPage:
    """Parse the raw olmOCR model output into an :class:`OCRPage`.

    The model produces::

        ---
        primary_language: en
        is_rotation_valid: True
        rotation_correction: 0
        is_table: False
        is_diagram: False
        ---
        <markdown content>

    We extract structured regions (tables, formulas, section headers, images)
    where possible, and always populate ``full_text`` with the readable body.
    """
    metadata, body = parse_yaml_frontmatter(raw_output)

    regions: list[OCRRegion] = []
    order_idx = 0

    # --- Tables (Markdown pipe-tables or HTML) ---
    # Markdown pipe-tables: lines starting with |
    md_table_pattern = re.compile(r"((?:^\|.+\|$\n?){2,})", re.MULTILINE)
    for match in md_table_pattern.finditer(body):
        regions.append(
            OCRRegion(
                text=match.group(1).strip(),
                category="table",
                text_format="markdown",
                order=order_idx,
                confidence=-1.0,
            )
        )
        order_idx += 1

    # HTML tables (less common in olmOCR output, but handle anyway)
    html_table_pattern = re.compile(r"(<table[\s\S]*?</table>)", re.IGNORECASE)
    for match in html_table_pattern.finditer(body):
        regions.append(
            OCRRegion(
                text=match.group(1).strip(),
                category="table",
                text_format="html",
                order=order_idx,
                confidence=-1.0,
            )
        )
        order_idx += 1

    # --- LaTeX formulas (display math) ---
    formula_patterns = [
        re.compile(r"\$\$(.+?)\$\$", re.DOTALL),
        re.compile(r"\\\[(.+?)\\\]", re.DOTALL),
    ]
    for pat in formula_patterns:
        for match in pat.finditer(body):
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

    # --- Section headers (Markdown ## style) ---
    header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    for match in header_pattern.finditer(body):
        level = len(match.group(1))
        cat = "title" if level == 1 else "section-header"
        regions.append(
            OCRRegion(
                text=match.group(2).strip(),
                category=cat,
                text_format="markdown",
                order=order_idx,
                confidence=-1.0,
            )
        )
        order_idx += 1

    return OCRPage(
        full_text=body,
        regions=regions,
        width=width,
        height=height,
    )


# ---------------------------------------------------------------------------
# Worker entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry-point invoked by the benchmark runner subprocess protocol."""
    parser = argparse.ArgumentParser(description="olmOCR-2 worker")
    parser.add_argument("--task", required=True, help="Path to input task JSON")
    parser.add_argument("--output", required=True, help="Path for output JSON")
    args = parser.parse_args()

    # Load task
    with open(args.task, "r", encoding="utf-8") as f:
        task = WorkerTask.from_dict(json.load(f))

    # Resolve device
    device = task.device or "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        logger.warning("CUDA not available, falling back to CPU")

    # Allow overriding model/processor via task params
    model_id = task.params.get("model_id", DEFAULT_MODEL_ID)
    processor_id = task.params.get("processor_id", DEFAULT_PROCESSOR_ID)
    target_longest_dim = int(
        task.params.get("target_longest_image_dim", TARGET_LONGEST_IMAGE_DIM)
    )
    temperature = float(task.params.get("temperature", 0.1))
    max_new_tokens = int(task.params.get("max_new_tokens", 16384))

    # ── Memory tracking: before load ──
    ram_before = get_ram_mb()
    vram_before = get_vram_usage_mb()
    reset_peak_vram()

    # ── Load model ──
    t0 = time.perf_counter()
    model = OlmOCRModel(
        model_id=model_id,
        processor_id=processor_id,
        device=device,
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
                target_longest_dim=target_longest_dim,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            pred_time = time.perf_counter() - t1

            logger.debug(
                f"Raw output ({len(raw_output)} chars): " f"{raw_output[:500]}..."
            )

            ocr_page = parse_olmocr_output(raw_output, width, height)
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
