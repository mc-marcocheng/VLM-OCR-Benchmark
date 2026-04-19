# VLM OCR Benchmark

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

```text
ocr-benchmark/
├── pyproject.toml                  # Root workspace (uv)
├── config/
│   └── default.yaml                # Full default configuration
├── packages/
│   └── ocr-core/                   # Shared framework library
├── models/                         # Isolated model projects (NOT workspace members)
│   ├── model_glm_ocr/              # GLM-OCR
│   ├── model_dots_mocr/            # Dots.MOCR
│   └── huggingface_cache/          # Shared weight cache
├── scripts/
│   ├── run_benchmark.py            # CLI entry-point
│   └── convert_label_studio.py     # Label Studio export converter
├── app.py                          # Gradio dashboard
├── data/
│   ├── inputs/<test_set>/          # Input PDFs and images
│   ├── groundtruths/<test_set>/    # Ground truth (.json or .txt)
│   └── processed/<test_set>/       # Cached page images
└── results/                        # Output (JSON and summary.csv)
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
git clone https://github.com/mc-marcocheng/VLM-OCR-Benchmark && cd VLM-OCR-Benchmark

# Install the root workspace (ocr-core + dashboard dependencies)
uv sync

# Install each model environment (downloads its own dependencies)
uv sync --directory models/model_glm_ocr
uv sync --directory models/model_dots_mocr
```

**Optional Extras:**

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
uv run scripts/gradio_app.py
```

You can also customize the launch parameters (e.g., to run on a specific port, share publicly, or use a custom config):
```bash
uv run scripts/gradio_app.py --host 0.0.0.0 --port 8080 --share --config config/default.yaml
```

Opens a Gradio interface at `http://127.0.0.1:7860` (or your specified host/port) with three tabs:

- **📊 Summary** — table of all benchmark runs from `results/summary.csv`.
- **🔎 Explorer** — select a test set, file, model, and device to inspect
  per-page images, predictions, ground truth, and metric scores.
- **⚖️ Compare** — side-by-side output from two model/device combinations on
  the same input file.

### Convert Label Studio Annotations

Converts a Label Studio JSON export into the benchmark's ground-truth format. It automatically reads your `config.yaml` to save the output in the correct directory.

```bash
uv run scripts/convert_label_studio.py <export.json> <test_set> [--config path/to/config.yaml]

# Examples:
uv run scripts/convert_label_studio.py project-1-at-2026-01-01.json test_1
uv run scripts/convert_label_studio.py project-1-at-2026-01-01.json test_1 --config custom_config.yaml
```

---

## Preparing Data

### Input Files

Place PDF or image files under `data/inputs/<test_set>/`:

```text
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
      "full_text": "Invoice #1234\nDate: 2026-01-01\n...",
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

```text
First page text goes here.
[PAGE_BREAK]
Second page text goes here.
```

### Region Categories & Formats
- **Categories:** `text`, `title`, `section-header`, `caption`, `footnote`, `page-header`, `page-footer`, `list-item`, `table`, `formula`, `picture`.
- **Text Formats:** `plain`, `markdown`, `latex`, `html` (e.g., TEDS expects `html` for tables).

---

## Configuration

All settings live in `config/default.yaml`. Every section is optional — the framework uses sensible defaults.

### Metrics

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

*(Metrics that require specific data are automatically skipped when the ground truth or prediction lacks that data).*

### Normalisation & Degradation Examples
*(See `config/default.yaml` for a full list of settings)*

```yaml
normalisation:
  unicode_form: "NFKC"              # NFKC, NFC, NFD, NFKD, or ""
  lowercase: false
  strip_whitespace: true
  collapse_whitespace: true
  traditional_to_simplified: false  # requires opencc extra

degradation:
  enabled: false
  pipelines:
    - name: noise
      params: { sigma: [10, 25, 50] }
    - name: blur
      params: { radius: [1.0, 2.0, 4.0] }
```

---

## Output Format

### `results/summary.csv`
One row per (model, test set, device) combination with aggregated metrics, timing, and resource usage.

### `results/<test_set>/<model>_<device>_results.json`
Full detail including:
- Per-page predictions and ground truth text
- Per-page metric scores and predicted regions with bboxes
- Multi-run metric summaries with 95% confidence intervals
- Resource usage (RAM, VRAM, model load time)
- Degradation results (when enabled)

---

## Contributing

Want to add a new model, custom metric, or help improve the framework? Check out our [Contributing Guide](CONTRIBUTING.md) for architectural details and step-by-step instructions.
