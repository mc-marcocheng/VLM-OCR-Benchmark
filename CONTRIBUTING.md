# Contributing to VLM OCR Benchmark

First off, thank you for considering contributing to the framework!

This guide outlines the architecture of the project and explains how to extend it by adding new OCR models or custom evaluation metrics.

## Development Setup

The project uses `uv` for lightning-fast package management and `pre-commit` for code formatting.

```bash
# Clone the repository
git clone https://github.com/mc-marcocheng/VLM-OCR-Benchmark && cd VLM-OCR-Benchmark

# Install the dev environment (installs pre-commit)
uv sync --extra dev

# Setup pre-commit hooks
uv run pre-commit install
```
Code formatting relies on `black` and `isort`. Ensure your editor is configured to use them, or rely on `pre-commit` to format files before commit.

---

## Architecture

### Why Model Isolation?

OCR models often pin incompatible versions of large dependencies (`transformers`, `torch`, `flash-attn`, etc.). For example, one model might require `transformers==4.57.6` while another needs `transformers>=5.5.4`.

The framework solves this by keeping each model in its own `uv` project **outside the root workspace**:

```text
Root workspace (pyproject.toml)
├── [tool.uv.workspace] members = ["packages/*"]   ← ocr-core only
│
├── packages/ocr-core/          ← workspace member (core framework)
│
├── models/model_glm_ocr/       ← NOT a workspace member
│   └── pyproject.toml          ← isolated env, depends on ocr-core via path:
│       [tool.uv.sources]
│       ocr-core = { path = "../../packages/ocr-core" }
│
└── models/model_dots_mocr/     ← NOT a workspace member (same pattern)
```

### Worker Protocol

The benchmark runner orchestrates evaluations by spawning each model as an isolated subprocess. They communicate via strictly typed JSON objects (`WorkerTask` and `WorkerResponse` defined in `ocr_core.types`).

```text
BenchmarkRunner                          Model Worker (isolated venv)
     │                                        │
     ├─ write task.json ─────────────────────►│
     │  (image paths, device, params)         │
     │                                        │
     ├─ uv run --directory <project_dir>      │
     │    python -m <module>                  │
     │    --task task.json                    │
     │    --output result.json                │
     │                                        │
     │◄─ read result.json ────────────────────┤
     │  (per-page OCRPage, timing, RAM/VRAM)  │
```

---

## Adding a New Model

To add a new model wrapper, you will create a new directory in `models/`, give it its own `pyproject.toml`, and write a `worker.py` script.

### 1. Create the Project Directory

```bash
mkdir -p models/model_myocr/src/model_myocr
touch models/model_myocr/src/model_myocr/__init__.py
touch models/model_myocr/src/model_myocr/worker.py
touch models/model_myocr/src/model_myocr/__main__.py
```

### 2. Write `pyproject.toml`
Create `models/model_myocr/pyproject.toml`. Ensure it depends on the local `ocr-core` package.

```toml
[project]
name = "model-myocr"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "ocr-core",
    # ... add your model-specific deps here (e.g. "transformers==4.30.0")
]

[tool.uv.sources]
ocr-core = { path = "../../packages/ocr-core" }

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/model_myocr"]
```

### 3. Implement `worker.py`
The worker needs to parse `WorkerTask`, run inference, track resource usage, and output a `WorkerResponse`. Here is a boilerplate implementation:

```python
"""
models/model_myocr/src/model_myocr/worker.py
"""
import argparse
import json
import time

from ocr_core.types import OCRPage, WorkerPageResult, WorkerResponse, WorkerTask
from ocr_core.utils import get_peak_vram_mb, reset_peak_vram, resolve_device
from loguru import logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # 1. Parse Task
    with open(args.task, encoding="utf-8") as f:
        task = WorkerTask.from_dict(json.load(f))

    device = resolve_device(task.device)
    reset_peak_vram()

    # 2. Load Model & Track Time
    t0 = time.perf_counter()
    # TODO: Load your specific model here
    # model, processor = load_my_model(device=device)
    model_load_time = time.perf_counter() - t0

    pages = []

    # 3. Process Images
    for img_path in task.image_paths:
        t0 = time.perf_counter()

        # TODO: Run Inference
        # text, regions = run_inference(img_path)

        # Construct the OCRPage result
        page_result = OCRPage(
            full_text="Extracted text here...",
            regions=[] # Populate with OCRRegion objects if layout detection is supported
        )

        pred_time = time.perf_counter() - t0

        pages.append(
            WorkerPageResult(
                image_path=img_path,
                prediction_time_seconds=pred_time,
                result=page_result,
            )
        )

    # 4. Construct Response & Write Output
    response = WorkerResponse(
        pages=pages,
        model_load_time_seconds=model_load_time,
        peak_vram_mb=get_peak_vram_mb(),
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(response.to_dict(), f, ensure_ascii=False, indent=2)

    logger.info(f"Worker done → {args.output}")

if __name__ == "__main__":
    main()
```

Set up the `__main__.py` entry point:
```python
"""models/model_myocr/src/model_myocr/__main__.py"""
from model_myocr.worker import main
main()
```

### 4. Register in Configuration
Add your model to the `models` block in `config/default.yaml`:

```yaml
models:
  MyOCR:
    project_dir: "models/model_myocr"
    module: "model_myocr.worker"
    params: {}  # Passed to WorkerTask.params
```

### 5. Install the environment
```bash
uv sync --directory models/model_myocr
```

---

## Adding a New Metric

Metrics are housed in `packages/ocr-core/src/ocr_core/metrics/`.

### 1. Create the Metric Class
Create a new file, e.g., `my_metric.py`:

```python
"""packages/ocr-core/src/ocr_core/metrics/my_metric.py"""
from ocr_core.metrics.base import Metric, MetricResult
from ocr_core.normalisation import NormalisationPipeline
from ocr_core.types import OCRPage

class MyMetric(Metric):
    name = "my_metric"            # Used in YAML config
    primary_key = "my_metric_f1"  # Used for aggregated reporting

    def is_applicable(self, gt_page: OCRPage, pred_page: OCRPage) -> bool:
        # Override this if your metric requires specific structured data (e.g., layout bboxes)
        # return len(gt_page.regions) > 0
        return True

    def compute(
        self,
        gt_page: OCRPage,
        pred_page: OCRPage,
        normaliser: NormalisationPipeline,
    ) -> MetricResult:
        # 1. Normalise text using the configured pipeline
        ref = normaliser.apply(gt_page.full_text)
        hyp = normaliser.apply(pred_page.full_text)

        # 2. Compute your metric logic
        score = 0.95

        # 3. Return a MetricResult with the scores dictionary
        return MetricResult(
            scores={"my_metric_f1": score},
            details={"extra_info": "Useful for debugging but not aggregated"}
        )
```

### 2. Register the Metric
Open `packages/ocr-core/src/ocr_core/metrics/registry.py` and add it to `_BUILTIN`:

```python
from ocr_core.metrics.my_metric import MyMetric

_BUILTIN: dict[str, type[Metric]] = {
    # ...
    "my_metric": MyMetric,
}
```

### 3. Add to Config
Enable it in `config/default.yaml`:

```yaml
metrics:
  # ... existing metrics
  - name: my_metric
    params: {}
```

---

## Working with AI Assistants

If you are using an AI assistant (like Claude, Cursor, or GitHub Copilot) to help develop, point it to `.claude/CLAUDE.md` and `models/.claude/CLAUDE.md` if available. These files contain specific project architecture rules and context designed to keep AI suggestions aligned with the framework's design philosophy.
