## What This Project Is

An OCR benchmarking framework that evaluates models across text-accuracy metrics,
layout metrics, and table-structure metrics. Models run in isolated virtual
environments (separate uv projects) to avoid dependency conflicts, and
communicate with the orchestrator via JSON over subprocess stdio.

## Repository Layout

```
pyproject.toml                          Root uv workspace
packages/ocr-core/                      Shared library (workspace member)
  src/ocr_core/
    types.py                            Canonical data types (OCRPage, BBox, etc.)
    config.py                           YAML config → typed dataclasses
    benchmark.py                        BenchmarkRunner — orchestrates everything
    data_loader.py                      PDF→images, ground truth loading
    normalisation.py                    Text normalisation pipeline
    degradation.py                      Image degradation for robustness
    statistics.py                       Bootstrap CI, paired tests
    reporting.py                        Markdown report generation
    utils.py                            VRAM helpers, formatting
    metrics/
      base.py                           Metric ABC + MetricResult
      registry.py                       Name→class registry
      cer.py wer.py char_f1.py          Text-accuracy metrics
      bleu.py bag_of_words.py           Text-accuracy metrics
      teds.py                           Table structure (HTML tree-edit distance)
      layout_iou.py                     Bounding-box IoU with Hungarian matching
      reading_order.py                  Kendall's τ on region order
models/                                 Isolated model projects (NOT workspace members)
  model_glm_ocr/                        GLM-OCR — transformers v5
  model_dots_mocr/                      Dots.MOCR — transformers v4
  huggingface_cache/                    Shared HF weight cache
config/default.yaml                     Full default configuration
scripts/run_benchmark.py                CLI entry-point
app.py                                  Gradio dashboard
data/inputs/ groundtruths/ processed/   Data directories
results/                                Output directory
```

## Key Commands

```bash
# Install root workspace
uv sync

# Install a model environment
uv sync --directory models/model_glm_ocr
uv sync --directory models/model_dots_mocr

# Run a benchmark
python scripts/run_benchmark.py --model GLMOCR --test_set test_1 --device cuda

# Launch dashboard
python app.py
```

There are no tests yet. When tests are added they will likely use `pytest` and
live in a `tests/` directory at the root.

## Architecture Decisions

### Model Isolation Pattern

Models live **outside** the uv workspace (`[tool.uv.workspace] members` only
includes `packages/*`). Each model has its own `pyproject.toml` with a path
dependency on ocr-core:

```toml
[tool.uv.sources]
ocr-core = { path = "../../packages/ocr-core" }
```

Models are invoked via:
```
uv run --directory <project_dir> python -m <module> --task <path> --output <path>
```

This is the fundamental mechanism that allows models with conflicting transitive
dependencies to coexist. Never add model directories to workspace members.

### Worker Protocol

Communication is via temporary JSON files:
- **Input**: `WorkerTask` — image paths, device, params
- **Output**: `WorkerResponse` — model load time, RAM/VRAM, list of
  `WorkerPageResult` each containing an `OCRPage`

All canonical types are in `ocr_core.types`. Workers must write valid JSON
conforming to `WorkerResponse.from_dict()`.

### Metric Applicability

Each metric has an `is_applicable(gt_page, pred_page) -> bool` method. Metrics
that require bboxes (layout_iou), tables (teds), or ordered regions
(reading_order) automatically skip pages that lack the required data. Text
metrics (cer, wer, char_f1, bleu, bag_of_words) always apply when text exists.

### jiwer API (v4+)

The project requires `jiwer>=4.0.0`. Use `process_words()` and
`process_characters()` — NOT the removed `compute_measures()`. Return types are
`WordOutput` and `CharacterOutput` with `.wer`/`.cer`, `.hits`,
`.substitutions`, `.deletions`, `.insertions` as integer/float attributes.

### Normalisation

`NormalisationPipeline.apply()` returns normalised text. For WER on CJK text,
use `tokenise_for_wer()` which returns a `str` with spaces around each CJK
character (so each character becomes a "word"). Do NOT iterate or re-join the
return value — pass it directly to `process_words()`.

### Statistics

`summarise(values)` returns a `SummaryStats` dataclass with bootstrap 95% CI.
`paired_bootstrap_test(a, b)` returns a p-value. Both use
`numpy.random.default_rng(42)` for reproducibility.

## Conventions

- **Type hints everywhere** — use `from __future__ import annotations` at the
  top of every module.
- **Dataclass serialisation** — types use `.to_dict()` / `.from_dict()` instead
  of external serialisation libraries.
- **Metric scores convention** — a score of `-1.0` means "not applicable" or
  "no reference text". This is filtered out during aggregation.
- **Logging** — use `loguru.logger` everywhere, never `print()` or stdlib
  `logging`.
- **Image handling** — always convert to RGB via `Image.open(path).convert("RGB")`.
- **Path handling** — use `os.path` (not `pathlib`) for consistency with the
  existing codebase.
- **Config field names** — YAML keys must match dataclass field names exactly
  (e.g. `project_dir` not `package`).
- **Gradio outputs** — every return path of a Gradio callback must return
  exactly the same number of values as there are output components.

## Common Pitfalls

1. **Don't add models to workspace members** — this causes uv dependency
   resolution conflicts between models with incompatible deps.

2. **Model source directory name must match the Python module name** — if
   config says `module: "model_foo.worker"`, the source layout must be
   `src/model_foo/__init__.py` and `src/model_foo/worker.py`, and
   `[tool.hatch.build.targets.wheel] packages` must be `["src/model_foo"]`.

3. **`CACHE_DIR` in workers** — resolve relative to `__file__`, not `os.getcwd()`,
   because `uv run --directory` changes the working directory.

4. **Temp file cleanup** — the benchmark runner creates temp files for worker
   communication and degradation images. Both are cleaned up in `finally`
   blocks. New code that creates temp files must do the same.

## How to Add a New Model

See **`models/.claude/CLAUDE.md`** for detailed step-by-step instructions, rules on isolation, and a full Python boilerplate script for `worker.py` including VRAM/RAM tracking.

## How to Add a New Metric

1. Create `packages/ocr-core/src/ocr_core/metrics/<name>.py` subclassing
   `Metric`.
2. Register in `_BUILTIN` dict in `metrics/registry.py`.
3. Export in `metrics/__init__.py`.
4. Add to `config/default.yaml` under `metrics:`.

## File Dependencies (import graph)

```
types.py          ← no internal imports (leaf)
config.py         ← no internal imports (leaf)
utils.py          ← no internal imports (leaf)
normalisation.py  ← config
statistics.py     ← no internal imports (leaf)
metrics/base.py   ← normalisation, types
metrics/*.py      ← base, normalisation, types
metrics/registry.py ← config, base, all metric modules
data_loader.py    ← types
degradation.py    ← no internal imports (leaf)
reporting.py      ← benchmark, metrics, statistics, utils
benchmark.py      ← config, data_loader, degradation, metrics, normalisation,
                     statistics, types  (top-level orchestrator)
__init__.py       ← re-exports from all modules
```

No circular imports exist. `benchmark.py` is the integration point.
