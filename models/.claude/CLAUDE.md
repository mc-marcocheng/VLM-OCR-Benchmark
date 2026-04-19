# OCR Model Worker Development Guide

This directory contains the individual OCR model adapters. Because models often require conflicting dependencies (e.g., `transformers` v4 vs v5, or specific `torch` versions), **each model must be an isolated `uv` project**.

## ⚠️ Golden Rules
1. **Do not add model projects to the root workspace.** The root `pyproject.toml` `[tool.uv.workspace] members` explicitly excludes this `models/` directory to prevent dependency resolution conflicts.
2. **Depend on `ocr-core` via relative path.**
3. **Implement the CLI subprocess protocol.**
4. **Use the Shared HuggingFace Cache.** To avoid downloading massive LLM/VLM weights multiple times across different environments, all workers must force HuggingFace to use `models/huggingface_cache`.

## 1. Project Structure
To create a new model named `model_myocr`, structure it like this:

```text
models/model_myocr/
├── pyproject.toml
└── src/
    └── model_myocr/
        ├── __init__.py
        ├── __main__.py      # (Optional) allows `python -m model_myocr`
        └── worker.py        # The CLI entry point
```

## 2. `pyproject.toml` Template
```toml
[project]
name = "model-myocr"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "ocr-core",
    # ... your specific dependencies (torch, transformers, etc.)
]

[tool.uv.sources]
ocr-core = { path = "../../packages/ocr-core" }

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/model_myocr"]
```

## 3. Worker Protocol
The orchestrator (`BenchmarkRunner`) spawns your model in a subprocess:
```bash
uv run --directory models/model_myocr python -m model_myocr.worker \
  --task /tmp/task_xxx.json \
  --output /tmp/result_xxx.json
```

It communicates purely via JSON files using types defined in `ocr_core.types`.

## 4. Worker Boilerplate
Use the canonical types and VRAM helpers provided by `ocr-core`.

**Important:** The cache environment variables (`HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`) *must* be set before importing `transformers` or `torch`.

```python
import argparse
import json
import os
import time

# ── Configure HuggingFace Cache ──────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))
CACHE_DIR = os.path.join(_REPO_ROOT, "models", "huggingface_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Set environment variables BEFORE importing transformers to strictly enforce cache location
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_HUB_CACHE"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
# ─────────────────────────────────────────────────────────────

import psutil
from loguru import logger

# (Your ML imports go here)
# import torch
# from transformers import AutoModel, AutoProcessor

from ocr_core.types import OCRPage, WorkerTask, WorkerResponse, WorkerPageResult
from ocr_core.utils import get_vram_usage_mb, get_peak_vram_mb, reset_peak_vram, resolve_device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Parse input task
    with open(args.task, "r", encoding="utf-8") as f:
        task = WorkerTask.from_dict(json.load(f))

    device = resolve_device(task.device)
    logger.info(f"Loading MyOCR on {device}...")

    # 1. Track Load Time & Initial Resources
    t0 = time.perf_counter()
    ram_before = psutil.Process().memory_info().rss / (1024 * 1024)
    vram_before = get_vram_usage_mb()

    # TODO: Load your model and processor here!
    # Remember to pass `cache_dir=CACHE_DIR` to from_pretrained if applicable
    # model = AutoModel.from_pretrained(..., cache_dir=CACHE_DIR)

    load_time = time.perf_counter() - t0
    ram_after = psutil.Process().memory_info().rss / (1024 * 1024)
    vram_after = get_vram_usage_mb()

    reset_peak_vram()

    # 2. Process Pages
    pages = []
    for path in task.image_paths:
        logger.info(f"Processing {path}")
        t_start = time.perf_counter()

        # TODO: Run inference
        # result_page = OCRPage(full_text="Extracted text...", regions=[])
        result_page = OCRPage(full_text="")

        pred_time = time.perf_counter() - t_start
        pages.append(WorkerPageResult(
            image_path=path,
            prediction_time_seconds=pred_time,
            result=result_page
        ))

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
        pages=pages
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(response.to_dict(), f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
```
