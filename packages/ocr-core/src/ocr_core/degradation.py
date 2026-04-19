"""
Image degradation pipeline for robustness testing.

Each degradation returns a *new* PIL Image — originals are never mutated.
"""

from __future__ import annotations

import io
import itertools
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image, ImageFilter
from loguru import logger

__all__ = [
    "DegradationPipeline",
    "DegradationVariant",
    "DEGRADATION_FUNCTIONS",
    "add_gaussian_noise",
    "apply_blur",
    "jpeg_compress",
    "rotate",
    "reduce_dpi",
    "salt_and_pepper",
]


# ── Individual degradations ─────────────────────────────────


def add_gaussian_noise(
    image: Image.Image, sigma: float = 25.0, seed: int | None = None
) -> Image.Image:
    if seed is None:
        seed = 42 + int(sigma * 1_000_000)
    arr = np.array(image, dtype=np.float32)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_blur(image: Image.Image, radius: float = 2.0) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def jpeg_compress(image: Image.Image, quality: int = 50) -> Image.Image:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def rotate(image: Image.Image, degrees: float = 5.0) -> Image.Image:
    return image.rotate(degrees, resample=Image.BICUBIC,
                        expand=False, fillcolor=(255, 255, 255))


def reduce_dpi(image: Image.Image, factor: float = 0.5) -> Image.Image:
    w, h = image.size
    small = image.resize((max(1, int(w * factor)), max(1, int(h * factor))),
                         Image.LANCZOS)
    return small.resize((w, h), Image.LANCZOS)


def salt_and_pepper(
    image: Image.Image, amount: float = 0.02, seed: int | None = None
) -> Image.Image:
    if seed is None:
        seed = 42 + int(amount * 1_000_000)
    arr = np.array(image)
    rng = np.random.default_rng(seed)
    mask = rng.random(arr.shape[:2])
    arr[mask < amount / 2] = 0
    arr[mask > 1 - amount / 2] = 255
    return Image.fromarray(arr)


# ── Registry ────────────────────────────────────────────────

DEGRADATION_FUNCTIONS: dict[str, Callable[..., Image.Image]] = {
    "noise": add_gaussian_noise,
    "blur": apply_blur,
    "jpeg": jpeg_compress,
    "rotate": rotate,
    "dpi_reduction": reduce_dpi,
    "salt_and_pepper": salt_and_pepper,
}


# ── Pipeline ────────────────────────────────────────────────


@dataclass
class DegradationVariant:
    """One specific degradation with fixed parameters."""
    name: str
    label: str  # human-readable, e.g. "noise_sigma=25"
    fn: Callable[..., Image.Image]
    params: dict[str, Any]

    def apply(self, image: Image.Image) -> Image.Image:
        return self.fn(image, **self.params)


class DegradationPipeline:
    """
    Expand degradation config into concrete ``DegradationVariant`` objects.

    Example config step::

        {"name": "noise", "params": {"sigma": [10, 25, 50]}}

    expands to three variants — one per sigma value.
    """

    def __init__(self, steps: Sequence[dict[str, Any]]):
        self.variants: list[DegradationVariant] = []
        for step in steps:
            name = step.get("name", "")
            fn = DEGRADATION_FUNCTIONS.get(name)
            if fn is None:
                logger.warning(f"Unknown degradation: {name!r}")
                continue
            params = step.get("params", {})
            self._expand(name, fn, params)

    def _expand(self, name: str, fn: Callable, params: dict) -> None:
        """Expand list-valued params into individual variants via Cartesian product."""
        list_keys = [k for k, v in params.items() if isinstance(v, list)]
        if not list_keys:
            label = f"{name}({'|'.join(f'{k}={v}' for k, v in params.items())})"
            self.variants.append(DegradationVariant(name, label, fn, dict(params)))
            return

        # Build value lists for each list-key; keep scalars as-is
        sweep_keys = list_keys
        sweep_values = [params[k] for k in sweep_keys]

        for combo in itertools.product(*sweep_values):
            fixed = {}
            for k, v in params.items():
                if k in sweep_keys:
                    fixed[k] = combo[sweep_keys.index(k)]
                else:
                    fixed[k] = v
            label_parts = [f"{k}={fixed[k]}" for k in sweep_keys]
            label = f"{name}_{'_'.join(label_parts)}"
            self.variants.append(DegradationVariant(name, label, fn, fixed))

    def __iter__(self):
        return iter(self.variants)

    def __len__(self):
        return len(self.variants)
