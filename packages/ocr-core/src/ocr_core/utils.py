"""VRAM helpers and miscellaneous utilities."""

from __future__ import annotations

import math
import os
import re
from typing import Optional

from loguru import logger

__all__ = [
    "fmt",
    "get_peak_vram_mb",
    "get_vram_usage_mb",
    "reset_peak_vram",
    "resolve_device",
    "safe_filename",
]


def safe_filename(s: str) -> str:
    """Sanitise a string for use in file/directory names."""
    return re.sub(r"[^\w\-.]", "_", s)


def _get_device_id() -> int:
    """Attempt to resolve the active GPU device index for accurate VRAM polling."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.current_device()
    except Exception:
        pass
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        try:
            return int(cvd.split(",")[0])
        except ValueError:
            pass
    return 0


def get_vram_usage_mb() -> Optional[float]:
    device_id = _get_device_id()
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(device_id) / (1024**2)
    except Exception:
        pass
    try:
        from pynvml import (
            nvmlDeviceGetComputeRunningProcesses,
            nvmlDeviceGetHandleByIndex,
            nvmlInit,
            nvmlShutdown,
        )

        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(device_id)
        pid = os.getpid()
        for p in nvmlDeviceGetComputeRunningProcesses(h):
            if p.pid == pid:
                nvmlShutdown()
                return p.usedGpuMemory / (1024**2)
        nvmlShutdown()
    except Exception:
        pass
    return None


def get_peak_vram_mb() -> Optional[float]:
    device_id = _get_device_id()
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated(device_id) / (1024**2)
    except Exception:
        pass
    return None


def reset_peak_vram() -> None:
    device_id = _get_device_id()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device_id)
    except Exception:
        pass


def resolve_device(requested: str) -> str:
    requested = requested.lower().strip()
    if requested in ("gpu", "cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except (ImportError, AttributeError):
            pass
        logger.warning("CUDA not available — falling back to CPU")
        return "cpu"
    return requested  # Return as-is for "cpu" or other values


def fmt(value, spec: str = ".4f", suffix: str = "") -> str:
    if value is None or (isinstance(value, (int, float)) and math.isnan(value)):
        return "N/A"
    try:
        return f"{value:{spec}}{suffix}"
    except (ValueError, TypeError):
        return str(value)
