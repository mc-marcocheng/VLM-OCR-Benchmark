# 🔍 OCR Benchmark

A general-purpose framework for evaluating and comparing OCR models with
comprehensive metrics, statistical rigor, and an interactive dashboard.

## Key Features

- **Model isolation** — each model runs in its own virtual environment, so
  models with incompatible dependencies (e.g. `transformers` v4 vs v5) coexist
  without conflict.
- **8 built-in metrics** — CER, WER, character F1, BLEU, bag-of-words, TEDS
  (table structure), layout IoU, and reading-order correlation.
- **Multi-run statistics** — bootstrap confidence intervals, paired significance
  tests, warmup runs.
- **Degradation robustness** — automated sweeps over noise, blur, JPEG
  compression, rotation, and DPI reduction.
- **CJK-aware normalisation** — configurable pipeline with Unicode
  normalisation, full-width→half-width, Traditional→Simplified Chinese, and
  per-character tokenisation for WER.
- **Structured + legacy ground truth** — JSON with regions/bboxes, or plain
  `.txt` with `[PAGE_BREAK]` separators.
- **Gradio dashboard** — summary table, per-page explorer, and side-by-side
  model comparison.

---

## Project Structure

```
ocr-benchmark/
├── pyproject.toml                  # Root workspace (uv)
├── config/
│   └── default.yaml                # Full default configuration
├── packages/
│   └── ocr-core/                   # Shared framework library
│       └── src/ocr_core/
│           ├── benchmark.py        # Orchestrator
│           ├── config.py           # YAML configuration loader
│           ├── data_loader.py      # PDF/image ingestion + GT loading
│           ├── degradation.py      # Image degradation pipeline
│           ├── normalisation.py    # Text normalisation pipeline
│           ├── reporting.py        # Markdown report generation
│           ├── statistics.py       # Bootstrap CI + paired tests
│           ├── types.py            # Canonical data types
│           ├── utils.py            # VRAM helpers, formatting
│           └── metrics/            # All metric implementations
├── models/                         # Isolated model projects (NOT workspace members)
│   ├── model_glm_ocr/             # GLM-OCR
│   ├── model_dots_mocr/           # Dots.MOCR
│   └── huggingface_cache/         # Shared weight cache
├── scripts/
│   └── run_benchmark.py           # CLI entry-point
├── app.py                         # Gradio dashboard
├── data/
│   ├── inputs/<test_set>/         # Input PDFs and images
│   ├── groundtruths/<test_set>/   # Ground truth (.json or .txt)
│   └── processed/<test_set>/      # Cached page images
└── results/                       # Output
    ├── summary.csv
    └── <test_set>/<model>_<device>_results.json
```

---

## Prerequisites

- **Python ≥ 3.13**
- **[uv](https://docs.astral.sh/uv/)** — used for workspace management and
  isolated model environments.

Install uv if you haven't:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Setup

```bash
# Clone the repository
git clone <repo-url> && cd ocr-benchmark

# Install the root workspace (ocr-core + dashboard dependencies)
uv sync

# Install each model environment (downloads its own dependencies)
uv sync --directory models/model_glm_ocr
uv sync --directory models/model_dots_mocr
```

For GPU support, install the optional extras in ocr-core:

```bash
uv sync --extra gpu
```

For Traditional→Simplified Chinese normalisation:

```bash
uv sync --extra chinese
```

For optimal (Hungarian) matching in layout IoU metrics:

```bash
uv sync --extra optimal-matching
```

---

## Usage

### Run a Benchmark

```bash
# Basic run
uv run scripts/run_benchmark.py \
    --model GLMOCR \
    --test_set test_1 \
    --device cuda

# With custom config, extra runs, and degradation
uv run scripts/run_benchmark.py \
    --model DotsMOCR \
    --test_set test_1 \
    --device cuda \
    --config config/default.yaml \
    --runs 5 \
    --warmup 2 \
    --degradation

# Generate a Markdown report
uv run scripts/run_benchmark.py \
    --model GLMOCR \
    --test_set test_1 \
    --device cpu \
    --report results/report.md
```

**CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | Model name as defined in config YAML |
| `--test_set` | *(required)* | Folder name under `data/inputs/` |
| `--device` | `cpu` | `cpu`, `cuda`, or `gpu` (alias for `cuda`) |
| `--config` | `config/default.yaml` | Path to configuration file |
| `--runs` | from config | Override number of measured runs |
| `--warmup` | from config | Override number of warmup runs |
| `--degradation` | off | Enable degradation robustness testing |
| `--report` | none | Path to write a Markdown report |

### Launch the Dashboard

```bash
uv run app.py
```

Opens a Gradio interface at `http://localhost:7860` with three tabs:

- **📊 Summary** — table of all benchmark runs from `results/summary.csv`.
- **🔎 Explorer** — select a test set, file, model, and device to inspect
  per-page images, predictions, ground truth, and metric scores.
- **⚖️ Compare** — side-by-side output from two model/device combinations on
  the same input file.

### Convert Label Studio Annotations

```bash
uv run scripts/convert_label_studio.py <export.json> <test_set>

# Example:
uv run scripts/convert_label_studio.py project-1-at-2026-01-01.json test_1
```

---

## Preparing Data

### Input Files

Place PDF or image files under `data/inputs/<test_set>/`:

```
data/inputs/invoices/
├── invoice_001.pdf
├── invoice_002.png
└── receipt.jpg
```

Supported formats: `.pdf`, `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`,
`.heic`, `.heif`, `.webp`.

PDFs are automatically rasterised at the configured DPI (default 200) and cached
under `data/processed/`.

### Ground Truth

Place files under `data/groundtruths/<test_set>/` with the same stem as the
input file.

**Structured JSON** (recommended — enables layout and table metrics):

```json
{
  "source_file": "invoice_001.pdf",
  "pages": [
    {
      "page_number": 1,
      "full_text": "Invoice #1234\nDate: 2024-01-15\n...",
      "regions": [
        {
          "text": "Invoice #1234",
          "category": "title",
          "bbox": {"x1": 50, "y1": 20, "x2": 400, "y2": 60},
          "order": 0
        },
        {
          "text": "<table><tr><td>Item</td><td>Price</td></tr>...</table>",
          "category": "table",
          "text_format": "html",
          "bbox": {"x1": 50, "y1": 200, "x2": 500, "y2": 600},
          "order": 3
        }
      ]
    }
  ]
}
```

**Legacy plain text** (fallback — enables text-only metrics):

```
First page text goes here.
[PAGE_BREAK]
Second page text goes here.
```

### Region Categories

`text`, `title`, `section-header`, `caption`, `footnote`, `page-header`,
`page-footer`, `list-item`, `table`, `formula`, `picture`.

### Text Formats

`plain`, `markdown`, `latex`, `html` — the `text_format` field on regions tells
metrics how to interpret content (e.g. TEDS expects `html` for tables).

---

## Metrics

| Metric | Key(s) | Requires | Description |
|--------|--------|----------|-------------|
| **CER** | `cer` | text | Character Error Rate via edit distance |
| **WER** | `wer` | text | Word Error Rate (CJK-aware tokenisation) |
| **Char F1** | `char_precision`, `char_recall`, `char_f1` | text | Character-level precision, recall, F1 |
| **BLEU** | `bleu` | text | N-gram precision with brevity penalty |
| **Bag of Words** | `bow_precision`, `bow_recall`, `bow_f1`, `bow_jaccard` | text | Order-insensitive word overlap |
| **TEDS** | `teds` | table regions | Tree-Edit-Distance Similarity for HTML tables |
| **Layout IoU** | `layout_mean_iou`, `layout_precision`, `layout_recall` | bounding boxes | Hungarian-matched box IoU with category check |
| **Reading Order** | `reading_order_tau` | ordered regions | Kendall's τ on matched region indices |

Metrics that require specific data (bboxes, tables, ordered regions) are
automatically skipped when the ground truth or prediction lacks that data.

---

## Configuration

All settings live in `config/default.yaml`. Every section is optional — the
framework uses sensible defaults.

### Normalisation

```yaml
normalisation:
  unicode_form: "NFKC"              # NFKC, NFC, NFD, NFKD, or ""
  lowercase: false
  strip_whitespace: true
  collapse_whitespace: true
  remove_punctuation: false
  fullwidth_to_halfwidth: false      # ！→ !  etc.
  traditional_to_simplified: false   # requires opencc extra
  custom_replacements: {}            # {"old": "new", ...}
```

### Metrics

```yaml
metrics:
  - name: cer
  - name: wer
  - name: char_f1
  - name: bleu
    params:
      max_n: 4
  - name: bag_of_words
  - name: teds
  - name: layout_iou
    params:
      iou_threshold: 0.5
  - name: reading_order
```

### Benchmark

```yaml
benchmark:
  runs: 3            # Measured runs (results are aggregated)
  warmup_runs: 1     # Discarded warm-up runs
  timeout_seconds: 3600
```

### Degradation

```yaml
degradation:
  enabled: false
  pipelines:
    - name: noise
      params: { sigma: [10, 25, 50] }
    - name: blur
      params: { radius: [1.0, 2.0, 4.0] }
    - name: jpeg
      params: { quality: [30, 50, 75] }
    - name: rotate
      params: { degrees: [-5, -2, 2, 5] }
    - name: dpi_reduction
      params: { factor: [0.5, 0.25] }
```

List-valued parameters are expanded into one degradation variant per value.

### Models

```yaml
models:
  GLMOCR:
    project_dir: "models/model_glm_ocr"
    module: "model_glm_ocr.worker"
    params: {}
  DotsMOCR:
    project_dir: "models/model_dots_mocr"
    module: "model_dots_mocr.worker"
    params: {}
```

- `project_dir` — path to the model's uv project (relative to repo root).
- `module` — Python module invoked via `uv run python -m <module>`.
- `params` — arbitrary dict forwarded to the worker.

---

## Architecture

### Why Model Isolation?

OCR models often pin incompatible versions of large dependencies
(`transformers`, `torch`, etc.). The framework solves this by keeping each model
in its own uv project **outside the workspace**:

```
Root workspace (pyproject.toml)
├── [tool.uv.workspace] members = ["packages/*"]   ← ocr-core only
│
├── packages/ocr-core/          ← workspace member
│
├── models/model_glm_ocr/       ← NOT a workspace member
│   └── pyproject.toml           ← depends on ocr-core via path
│       [tool.uv.sources]
│       ocr-core = { path = "../../packages/ocr-core" }
│
└── models/model_dots_mocr/      ← NOT a workspace member
    └── pyproject.toml           ← same pattern
```

### Worker Protocol

The benchmark runner spawns each model as a subprocess:

```
BenchmarkRunner                          Model Worker
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

Workers receive a `WorkerTask` JSON and write a `WorkerResponse` JSON. The
canonical types are defined in `ocr_core.types`.

### Statistical Pipeline

1. **Warmup runs** are executed and discarded.
2. **Measured runs** are aggregated per-metric using bootstrap CIs.
3. **Degradation runs** (optional) sweep parameter grids and report metric
   deltas.
4. `BenchmarkRunner.compare()` performs paired bootstrap significance testing
   between two results.

---

## Contributing

### AI Assistant Guides

If you are using an AI assistant (like Claude or GitHub Copilot) to help develop, point it to the `.claude/CLAUDE.md` and `models/.claude/CLAUDE.md` files which contain architecture rules and boilerplate.

### Adding a New Model

1. Create a project directory:

   ```
   models/model_myocr/
   ├── pyproject.toml
   └── src/model_myocr/
       ├── __init__.py
       └── worker.py
   ```

2. Write `pyproject.toml`:

   ```toml
   [project]
   name = "model-myocr"
   version = "0.1.0"
   requires-python = ">=3.13"
   dependencies = [
       "ocr-core",
       # ... model-specific deps
   ]

   [tool.uv.sources]
   ocr-core = { path = "../../packages/ocr-core" }

   [build-system]
   requires = ["hatchling"]
   build-backend = "hatchling.build"

   [tool.hatch.build.targets.wheel]
   packages = ["src/model_myocr"]
   ```

3. Implement the worker in `src/model_myocr/worker.py`:
   *(See `models/.claude/CLAUDE.md` for a complete boilerplate with VRAM tracking.)*

   ```python
   import argparse, json, time, psutil
   from ocr_core.types import OCRPage, WorkerTask, WorkerResponse, WorkerPageResult

   def main():
       parser = argparse.ArgumentParser()
       parser.add_argument("--task", required=True)
       parser.add_argument("--output", required=True)
       args = parser.parse_args()

       with open(args.task) as f:
           task = WorkerTask.from_dict(json.load(f))

       # Track load time...
       t0 = time.perf_counter()
       # model = load_model(...)
       load_time = time.perf_counter() - t0

       pages = []
       for img_path in task.image_paths:
           t0 = time.perf_counter()
           # result_page = predict(img_path)
           pred_time = time.perf_counter() - t0
           pages.append(WorkerPageResult(
               image_path=img_path, prediction_time_seconds=pred_time, result=result_page
           ))

       response = WorkerResponse(pages=pages, model_load_time_seconds=load_time)
       with open(args.output, "w") as f:
           json.dump(response.to_dict(), f)

   if __name__ == "__main__":
       main()
   ```

4. Register in `config/default.yaml`:

   ```yaml
   models:
     MyOCR:
       project_dir: "models/model_myocr"
       module: "model_myocr.worker"
       params: {}
   ```

5. Install:

   ```bash
   uv sync --directory models/model_myocr
   ```

## Adding a New Metric

1. Create `packages/ocr-core/src/ocr_core/metrics/my_metric.py`:

   ```python
   from ocr_core.metrics.base import Metric, MetricResult
   from ocr_core.normalisation import NormalisationPipeline
   from ocr_core.types import OCRPage

   class MyMetric(Metric):
       name = "my_metric"

       def is_applicable(self, gt_page, pred_page) -> bool:
           return True  # override to restrict

       def compute(self, gt_page: OCRPage, pred_page: OCRPage,
                   normaliser: NormalisationPipeline) -> MetricResult:
           # Your computation
           return MetricResult(scores={"my_metric": 0.95})
   ```

2. Register in `packages/ocr-core/src/ocr_core/metrics/registry.py`:

   ```python
   from ocr_core.metrics.my_metric import MyMetric
   _BUILTIN["my_metric"] = MyMetric
   ```

3. Add to config:

   ```yaml
   metrics:
     - name: my_metric
       params: {}
   ```

---

## Output Format

### `results/summary.csv`

One row per (model, test set, device) combination with aggregated metrics,
timing, and resource usage.

### `results/<test_set>/<model>_<device>_results.json`

Full detail including:

- Per-page predictions and ground truth text
- Per-page metric scores
- Predicted regions with bboxes (when available)
- Multi-run metric summaries with 95% confidence intervals
- Timing summary
- Resource usage (RAM, VRAM, model load time)
- Degradation results (when enabled)
