"""
Statistical helpers: bootstrap confidence intervals, paired tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = ["SummaryStats", "summarise", "bootstrap_ci", "paired_bootstrap_test"]


@dataclass
class SummaryStats:
    mean: float
    std: float
    median: float
    ci_lower: float  # 95 % CI
    ci_upper: float
    min: float
    max: float
    n: int


def summarise(values: Sequence[float], confidence: float = 0.95) -> SummaryStats:
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return SummaryStats(0, 0, 0, 0, 0, 0, 0, 0)
    if n == 1:
        v = float(arr[0])
        return SummaryStats(v, 0.0, v, v, v, v, v, 1)

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    median = float(np.median(arr))

    lo, hi = bootstrap_ci(arr, confidence=confidence)

    return SummaryStats(
        mean=mean,
        std=std,
        median=median,
        ci_lower=lo,
        ci_upper=hi,
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        n=n,
    )


def bootstrap_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
    statistic: str = "mean",
) -> tuple[float, float]:
    """Non-parametric bootstrap confidence interval for *statistic*."""
    rng = np.random.default_rng(42)
    n = len(values)

    # Vectorised: generate all bootstrap samples at once
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    samples = values[indices]

    if statistic == "mean":
        boot = samples.mean(axis=1)
    else:
        boot = np.median(samples, axis=1)

    alpha = (1 - confidence) / 2
    lo = float(np.percentile(boot, 100 * alpha))
    hi = float(np.percentile(boot, 100 * (1 - alpha)))
    return lo, hi


def paired_bootstrap_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_bootstrap: int = 10_000,
) -> float:
    """
    Two-sided paired bootstrap test.

    Returns the p-value for the null hypothesis that the mean of
    *scores_a* and *scores_b* are equal.  Lower is more significant.
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if len(a) != len(b):
        raise ValueError(
            f"Paired test requires equal-length arrays, got {len(a)} and {len(b)}"
        )

    observed_diff = abs(np.mean(a) - np.mean(b))
    diffs = a - b
    centred = diffs - np.mean(diffs)

    rng = np.random.default_rng(42)
    n = len(centred)

    # Vectorised: generate all bootstrap samples at once
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    samples = centred[indices]
    boot_means = np.abs(samples.mean(axis=1))

    return float(np.mean(boot_means >= observed_diff))
