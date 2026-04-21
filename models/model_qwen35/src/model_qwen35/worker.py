from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import time

import httpx
import psutil
from loguru import logger

from ocr_core.types import OCRPage, WorkerPageResult, WorkerResponse, WorkerTask

# ---------------------------------------------------------------------------
# Paths (resolved relative to __file__, not cwd)
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_MODELS_DIR = os.path.abspath(os.path.join(_MODEL_DIR, ".."))
CACHE_DIR = os.path.join(_MODELS_DIR, "huggingface_cache")
CHAT_TEMPLATE_PATH = os.path.join(_MODEL_DIR, "qwen35-chat-template.jinja")
COMPOSE_PATH = os.path.join(_MODEL_DIR, "docker-compose.generated.yml")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "model_id": "unsloth/Qwen3.5-27B-GGUF",
    "model_file": "Qwen3.5-27B-Q4_K_M.gguf",
    "mmproj_repo": "",
    "mmproj_file": "mmproj-BF16.gguf",
    "port": 8000,
    "ctx_size": 32768,
    "parallel": 1,
    "batch_size": 2048,
    "ubatch_size": 512,
    "enable_thinking": True,
    "timeout": 300,
    "max_tokens": 16384,
    "temperature": 0.1,
    "server_startup_timeout": 600,
    "keep_server": False,
    "force_restart": True,  # restart running container for clean VRAM baseline
    "vram_settle_seconds": 8,  # wait after stop for GPU memory to free
    "container_name": "ocr-qwen35-llama-cpp",
    "docker_image": "ghcr.io/ggml-org/llama.cpp:server-cuda13",
}
_MAX_RETRIES = 3
_RETRY_DELAY = 5

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
OCR_SYSTEM_PROMPT = (
    "You are a precise OCR (Optical Character Recognition) engine. "
    "Your sole task is to extract every piece of visible text from the "
    "provided image and reproduce it exactly as it appears.\n"
    "\n"
    "Strict rules:\n"
    "1. Reproduce text character-for-character — preserve spelling, "
    "punctuation, capitalisation, diacritics, accents, and whitespace.\n"
    "2. Maintain the original reading order: top-to-bottom, then "
    "left-to-right for LTR scripts or right-to-left for RTL scripts.\n"
    "3. Separate distinct paragraphs with exactly one blank line.\n"
    "4. For tables: render each row on its own line, separating columns "
    'with " | " (space-pipe-space). Preserve header rows as-is.\n'
    "5. For lists: preserve bullet characters, numbering, and indentation.\n"
    "6. Reproduce ALL languages exactly as shown — never translate or "
    "transliterate.\n"
    "7. Do NOT add commentary, interpretation, summary, labels, or "
    "metadata not visible in the image.\n"
    "8. Do NOT wrap output in markdown code blocks, quotation marks, or "
    "any other container.\n"
    "9. If no text is visible, respond with an empty string.\n"
    "10. Output ONLY the extracted text — nothing else."
)

OCR_USER_PROMPT = (
    "Extract all visible text from this image. "
    "Output the raw text only, preserving the original layout and reading order."
)


# ═══════════════════════════════════════════════════════════════════════════
# GPU monitoring  (nvidia-smi — host-wide, sees Docker container VRAM)
# ═══════════════════════════════════════════════════════════════════════════


def _gpu_vram_used_mb() -> float:
    """Total GPU memory used across all devices via nvidia-smi."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode == 0:
            return sum(float(x) for x in r.stdout.strip().split("\n") if x.strip())
    except Exception:
        pass
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# mmproj download
# ═══════════════════════════════════════════════════════════════════════════


def _ensure_mmproj(repo_id: str, filename: str) -> str:
    """Download the multimodal projector GGUF if not already cached.

    Returns the absolute path to the local file.
    """
    safe_repo = repo_id.replace("/", "--")
    mmproj_dir = os.path.join(CACHE_DIR, "gguf-mmproj", safe_repo)
    os.makedirs(mmproj_dir, exist_ok=True)
    local_path = os.path.join(mmproj_dir, filename)

    if os.path.isfile(local_path):
        logger.info("mmproj already cached: {}", local_path)
        return local_path

    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    tmp_path = local_path + ".downloading"
    logger.info("Downloading mmproj: {} → {}", url, local_path)

    try:
        with httpx.stream(
            "GET",
            url,
            follow_redirects=True,
            timeout=httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0),
        ) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            last_log = 0
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0 and downloaded - last_log >= 50 * 1024 * 1024:
                        logger.info(
                            "  {:.0f}% ({:.0f} / {:.0f} MB)",
                            downloaded / total * 100,
                            downloaded / 1e6,
                            total / 1e6,
                        )
                        last_log = downloaded

        os.replace(tmp_path, local_path)
        logger.info("Download complete: {} ({:.0f} MB)", local_path, downloaded / 1e6)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    return local_path


# ═══════════════════════════════════════════════════════════════════════════
# Docker lifecycle
# ═══════════════════════════════════════════════════════════════════════════

_COMPOSE_TEMPLATE = """\
# Auto-generated by model_qwen35 worker — do not edit
services:
  llama-cpp:
    container_name: {container_name}
    image: {docker_image}
    command: >
      --hf-repo {model_id}
      --hf-file {model_file}
      --mmproj /app/mmproj.gguf
      --jinja
      --port 8000
      --host 0.0.0.0
      --parallel {parallel}
      --batch-size {batch_size}
      --ubatch-size {ubatch_size}
      --cache-type-k q8_0
      --cache-type-v q8_0
      --flash-attn on
      --context-shift
      --ctx-size {ctx_size}
      --temp 0.1
      --top-k 20
      --chat-template-file /app/chat-template.jinja
      --reasoning on
    ports:
      - "{port}:8000"
    ipc: host
    environment:
      - HF_HOME=/root/.cache/huggingface
    volumes:
      - {cache_dir}:/root/.cache/huggingface/hub
      - {chat_template_path}:/app/chat-template.jinja:ro
      - {mmproj_path}:/app/mmproj.gguf:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
"""


def _generate_compose(cfg: dict, mmproj_path: str) -> str:
    """Write a docker-compose YAML and return its path."""
    content = _COMPOSE_TEMPLATE.format(
        container_name=cfg["container_name"],
        docker_image=cfg["docker_image"],
        model_id=cfg["model_id"],
        model_file=cfg["model_file"],
        parallel=cfg["parallel"],
        batch_size=cfg["batch_size"],
        ubatch_size=cfg["ubatch_size"],
        ctx_size=cfg["ctx_size"],
        port=cfg["port"],
        cache_dir=CACHE_DIR,
        chat_template_path=CHAT_TEMPLATE_PATH,
        mmproj_path=mmproj_path,
    )
    with open(COMPOSE_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("Generated compose file: {}", COMPOSE_PATH)
    return COMPOSE_PATH


def _is_container_running(name: str) -> bool:
    try:
        r = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", name],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return r.returncode == 0 and "true" in r.stdout.strip().lower()
    except Exception:
        return False


def _start_server(compose_path: str) -> None:
    logger.info("Starting llama.cpp container …")
    r = subprocess.run(
        ["docker", "compose", "-f", compose_path, "up", "-d"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if r.returncode != 0:
        logger.error(
            "docker compose up failed:\nstdout: {}\nstderr: {}", r.stdout, r.stderr
        )
        raise RuntimeError(
            f"docker compose up failed (exit {r.returncode}): {r.stderr}"
        )
    logger.info("Container started.")


def _stop_server(compose_path: str) -> None:
    logger.info("Stopping llama.cpp container …")
    subprocess.run(
        ["docker", "compose", "-f", compose_path, "down"],
        capture_output=True,
        text=True,
        timeout=60,
    )


def _wait_for_container_stop(name: str, timeout: float = 30) -> None:
    """Poll until the container is no longer running."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _is_container_running(name):
            logger.info("Container '{}' stopped.", name)
            return
        time.sleep(1)
    logger.warning("Container '{}' did not stop within {:.0f}s", name, timeout)


def _wait_for_health(
    base_url: str,
    timeout: float = 600,
    poll_interval: float = 5.0,
) -> None:
    """Block until llama.cpp /health reports ``{"status":"ok"}``."""
    health_url = base_url.rstrip("/") + "/health"
    deadline = time.monotonic() + timeout
    attempt = 0

    logger.info("Waiting for server at {} (timeout {:.0f}s) …", health_url, timeout)

    while time.monotonic() < deadline:
        attempt += 1
        try:
            resp = httpx.get(health_url, timeout=5)
            if resp.status_code == 200:
                body = resp.json()
                status = body.get("status", "")
                if status == "ok":
                    logger.info("Server healthy after {} polls.", attempt)
                    return
                logger.debug("Health status: {}", status)
            else:
                logger.debug("Health HTTP {}", resp.status_code)
        except httpx.HTTPError as exc:
            # Covers ConnectError, ReadTimeout, RemoteProtocolError,
            # ConnectTimeout, ReadError, CloseError, etc.
            logger.debug("Health poll {}: {}: {}", attempt, type(exc).__name__, exc)
        except OSError as exc:
            # Socket-level errors (connection refused before httpx wraps it)
            logger.debug("Health poll {}: OSError: {}", attempt, exc)

        remaining = deadline - time.monotonic()
        if attempt % 12 == 0:
            logger.info("  still waiting … {:.0f}s remaining", remaining)
        time.sleep(poll_interval)

    raise TimeoutError(f"llama.cpp server did not become healthy within {timeout}s")


# ═══════════════════════════════════════════════════════════════════════════
# HTTP helpers
# ═══════════════════════════════════════════════════════════════════════════


def _b64_encode(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _mime_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
        ".gif": "image/gif",
    }.get(ext, "image/png")


def _clean_response(text: str) -> str:
    """Strip thinking residue and formatting artefacts from model output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    if "</think>" in text:
        text = text.split("</think>", 1)[1]

    if "<think>" in text:
        text = text.split("<think>", 1)[0]

    text = re.sub(r"^```[^\n]*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)

    return text.strip()


def _post_with_retry(
    endpoint: str,
    payload: dict,
    timeout: float,
    max_retries: int = _MAX_RETRIES,
) -> str:
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = httpx.post(endpoint, json=payload, timeout=timeout)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return content or ""
        except (
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
        ) as exc:
            last_exc = exc
            if attempt < max_retries:
                logger.warning(
                    "Transient error (attempt {}/{}): {} — retry in {}s …",
                    attempt,
                    max_retries,
                    exc,
                    _RETRY_DELAY,
                )
                time.sleep(_RETRY_DELAY)
            else:
                logger.error("All {} attempts exhausted.", max_retries)
        except httpx.HTTPStatusError as exc:
            if 400 <= exc.response.status_code < 500:
                logger.error(
                    "Client error {} — not retrying: {}",
                    exc.response.status_code,
                    exc.response.text[:500],
                )
                raise
            last_exc = exc
            if attempt < max_retries:
                logger.warning(
                    "Server error {} (attempt {}/{}): retry in {}s …",
                    exc.response.status_code,
                    attempt,
                    max_retries,
                    _RETRY_DELAY,
                )
                time.sleep(_RETRY_DELAY)
            else:
                logger.error("All {} attempts exhausted.", max_retries)

    raise last_exc  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def _resolve(params: dict, key: str) -> object:
    return params.get(key, _DEFAULTS[key])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen3.5 OCR worker (llama.cpp Docker backend)",
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.task, "r", encoding="utf-8") as f:
        task = WorkerTask.from_dict(json.load(f))

    params = task.params or {}

    # ── Resolve configuration ─────────────────────────────────
    model_id = str(_resolve(params, "model_id"))
    model_file = str(_resolve(params, "model_file"))
    mmproj_repo = str(_resolve(params, "mmproj_repo")) or model_id
    mmproj_file = str(_resolve(params, "mmproj_file"))
    port = int(_resolve(params, "port"))
    ctx_size = int(_resolve(params, "ctx_size"))
    parallel = int(_resolve(params, "parallel"))
    batch_size = int(_resolve(params, "batch_size"))
    ubatch_size = int(_resolve(params, "ubatch_size"))
    enable_thinking = bool(_resolve(params, "enable_thinking"))
    timeout = float(_resolve(params, "timeout"))
    max_tokens = int(_resolve(params, "max_tokens"))
    temperature = float(_resolve(params, "temperature"))
    startup_timeout = float(_resolve(params, "server_startup_timeout"))
    keep_server = bool(_resolve(params, "keep_server"))
    force_restart = bool(_resolve(params, "force_restart"))
    settle_seconds = float(_resolve(params, "vram_settle_seconds"))
    container_name = str(_resolve(params, "container_name"))
    docker_image = str(_resolve(params, "docker_image"))

    base_url = f"http://localhost:{port}"
    completions_url = f"{base_url}/v1/chat/completions"

    logger.info("═══ Qwen3.5 worker configuration ═══")
    logger.info("  model_id:        {}", model_id)
    logger.info("  model_file:      {}", model_file)
    logger.info("  mmproj:          {}:{}", mmproj_repo, mmproj_file)
    logger.info("  port:            {}", port)
    logger.info("  ctx_size:        {}", ctx_size)
    logger.info("  enable_thinking: {}", enable_thinking)
    logger.info("  timeout:         {}s", timeout)
    logger.info("  max_tokens:      {}", max_tokens)
    logger.info("  keep_server:     {}", keep_server)
    logger.info("  force_restart:   {}", force_restart)

    # ── Validate chat template ────────────────────────────────
    if not os.path.isfile(CHAT_TEMPLATE_PATH):
        raise FileNotFoundError(
            f"Chat template not found: {CHAT_TEMPLATE_PATH}\n"
            "Place qwen35-chat-template.jinja in the model_qwen35 "
            "project root."
        )

    # ── Ensure mmproj downloaded ──────────────────────────────
    os.makedirs(CACHE_DIR, exist_ok=True)
    mmproj_local = _ensure_mmproj(mmproj_repo, mmproj_file)

    # ── Generate docker-compose ───────────────────────────────
    compose_cfg = {
        "container_name": container_name,
        "docker_image": docker_image,
        "model_id": model_id,
        "model_file": model_file,
        "parallel": parallel,
        "batch_size": batch_size,
        "ubatch_size": ubatch_size,
        "ctx_size": ctx_size,
        "port": port,
    }
    compose_path = _generate_compose(compose_cfg, mmproj_local)

    # ══════════════════════════════════════════════════════════
    # Server lifecycle + VRAM measurement
    #
    # Three cases:
    #   A) Container not running         → cold start (clean baseline)
    #   B) Running + force_restart=True   → stop, measure baseline, restart
    #   C) Running + force_restart=False  → reuse (no VRAM delta available)
    # ══════════════════════════════════════════════════════════
    ram_before = psutil.Process().memory_info().rss / (1024 * 1024)
    server_was_running = _is_container_running(container_name)

    if server_was_running and not force_restart:
        # ── Case C: warm reuse ────────────────────────────────
        logger.info(
            "Container '{}' already running — reusing.  "
            "Set force_restart=true for accurate VRAM delta.",
            container_name,
        )
        t0 = time.perf_counter()
        _wait_for_health(base_url, timeout=startup_timeout)
        load_time = time.perf_counter() - t0

        # Best we can do: snapshot current (model already loaded)
        vram_after = _gpu_vram_used_mb()
        # No clean baseline available — report 0 so the delta
        # (vram_after − vram_before) equals the full loaded footprint,
        # which is the most useful number a consumer can get.
        vram_before = 0.0
        logger.info(
            "Warm start — VRAM with model loaded: {:.0f} MB  "
            "(baseline unavailable, reported as 0)",
            vram_after,
        )
    else:
        # ── Case A / B: cold start (possibly after forced stop) ──
        if server_was_running:
            logger.info(
                "force_restart=true — stopping '{}' for clean VRAM baseline …",
                container_name,
            )
            _stop_server(compose_path)
            _wait_for_container_stop(container_name)
            logger.info(
                "Waiting {:.0f}s for GPU memory to settle …",
                settle_seconds,
            )
            time.sleep(settle_seconds)

        vram_before = _gpu_vram_used_mb()
        logger.info("VRAM baseline (no model): {:.0f} MB", vram_before)

        t0 = time.perf_counter()
        _start_server(compose_path)
        _wait_for_health(base_url, timeout=startup_timeout)
        load_time = time.perf_counter() - t0

        vram_after = _gpu_vram_used_mb()
        logger.info(
            "Cold start {:.1f}s — VRAM: {:.0f} → {:.0f} MB (Δ {:.0f} MB)",
            load_time,
            vram_before,
            vram_after,
            vram_after - vram_before,
        )

    ram_after = psutil.Process().memory_info().rss / (1024 * 1024)
    peak_vram = vram_after  # track max across inference

    # ── Process pages ─────────────────────────────────────────
    pages: list[WorkerPageResult] = []

    for img_path in task.image_paths:
        logger.info("Processing: {}", img_path)
        t_start = time.perf_counter()

        try:
            mime = _mime_type(img_path)
            b64 = _b64_encode(img_path)

            payload = {
                "messages": [
                    {"role": "system", "content": OCR_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{b64}",
                                },
                            },
                            {"type": "text", "text": OCR_USER_PROMPT},
                        ],
                    },
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "chat_template_kwargs": {
                    "enable_thinking": enable_thinking,
                },
            }

            raw_text = _post_with_retry(completions_url, payload, timeout)
            cleaned = _clean_response(raw_text)
            pred_time = time.perf_counter() - t_start

            pages.append(
                WorkerPageResult(
                    image_path=img_path,
                    prediction_time_seconds=pred_time,
                    result=OCRPage(full_text=cleaned, regions=[]),
                )
            )
            logger.info("  ✓ {:.1f}s — {} chars", pred_time, len(cleaned))

        except Exception as exc:
            pred_time = time.perf_counter() - t_start
            logger.error("  ✗ Failed on {}: {}", img_path, exc)
            pages.append(
                WorkerPageResult(
                    image_path=img_path,
                    prediction_time_seconds=pred_time,
                    result=OCRPage(full_text="", regions=[]),
                    error=str(exc),
                )
            )

        current_vram = _gpu_vram_used_mb()
        if current_vram > peak_vram:
            peak_vram = current_vram

    # ── Finalise ──────────────────────────────────────────────
    peak_ram = psutil.Process().memory_info().rss / (1024 * 1024)

    response = WorkerResponse(
        model_load_time_seconds=load_time,
        ram_before_load_mb=ram_before,
        ram_after_load_mb=ram_after,
        peak_ram_mb=peak_ram,
        vram_before_load_mb=vram_before,
        vram_after_load_mb=vram_after,
        peak_vram_mb=peak_vram,
        pages=pages,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(response.to_dict(), f, ensure_ascii=False, indent=2)

    logger.info("Results written to {}", args.output)

    # ── Tear down if requested ────────────────────────────────
    if not keep_server and not server_was_running:
        _stop_server(compose_path)
    elif not keep_server and server_was_running:
        logger.info(
            "keep_server=false but container was already running before "
            "this worker — leaving it up.",
        )


if __name__ == "__main__":
    main()
