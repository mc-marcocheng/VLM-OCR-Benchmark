# Qwen3.5-27B OCR Worker

Vision-language OCR using [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B)
quantised to Q4_K_M, served via **llama.cpp** in a Docker container.

## Architecture

The worker is a thin HTTP client that **manages the full server lifecycle**:

```
┌──────────────┐  1. download mmproj    ┌────────────────────┐
│              │  2. generate compose   │ huggingface_cache/ │
│              │  3. docker compose up  │   gguf-mmproj/…    │
│ worker.py    │  4. poll /health       └────────────────────┘
│ (subprocess) │  5. POST images ──────▶ ┌───────────────────┐
│              │  6. collect results ◀── │ llama.cpp Docker  │
│              │  7. (optional) down     │ (GPU, GGUF model) │
└──────────────┘                         └───────────────────┘
```

The worker monitors host-wide GPU VRAM via `nvidia-smi`, which sees the
container's memory usage through the passthrough GPU.

## Prerequisites

1. **Docker** with the `compose` plugin (`docker compose version`)
2. **NVIDIA Container Toolkit** (`nvidia-ctk --version`)
3. **Chat template** — place `qwen35-chat-template.jinja` in this directory

The worker automatically downloads `mmproj-BF16.gguf` on first run. The main
GGUF model is downloaded by llama.cpp inside the container (cached in the
shared `models/huggingface_cache`).

## Install & Run

```bash
uv sync --directory models/model_qwen35
uv run scripts/run_benchmark.py --model Qwen35 --test_set test_1
```

The first run will be slow (~5–15 min) as it downloads the model and loads
it into GPU memory. Subsequent runs reuse the running container.

## Configuration

```yaml
models:
  - name: Qwen35
    module: model_qwen35.worker
    project_dir: models/model_qwen35
    params:
      # ── Model selection ──
      model_id: "unsloth/Qwen3.5-27B-GGUF"     # HF repo
      model_file: "Qwen3.5-27B-Q4_K_M.gguf"    # GGUF quant file
      mmproj_repo: ""                            # empty = same as model_id
      mmproj_file: "mmproj-BF16.gguf"           # vision projector

      # ── Server tuning ──
      port: 8753
      ctx_size: 32768
      parallel: 1
      batch_size: 2048
      ubatch_size: 512
      docker_image: "ghcr.io/ggml-org/llama.cpp:server-cuda13"
      container_name: "ocr-qwen35-llama-cpp"

      # ── Inference ──
      enable_thinking: false        # Qwen3.5 reasoning mode
      timeout: 300                  # per-request timeout (s)
      max_tokens: 16384
      temperature: 0.1

      # ── Lifecycle ──
      server_startup_timeout: 600   # wait for model load (s)
      keep_server: true             # leave container running
```

### Swapping quant levels

```yaml
params:
  model_file: "Qwen3.5-27B-Q8_0.gguf"   # higher quality, more VRAM
```

### `enable_thinking`

When `true`, the model uses chain-of-thought reasoning before producing
output. Reasoning tokens are returned in a separate field by llama.cpp and
do not contaminate the OCR text. Expect 2–5× higher latency.

## Files

| File | Purpose |
|---|---|
| `qwen35-chat-template.jinja` | Jinja chat template mounted into container |
| `docker-compose.generated.yml` | Auto-generated on each run — do not edit |
| `src/model_qwen35/worker.py` | Worker entry point |
