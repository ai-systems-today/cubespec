"""Percentile bootstrap CI for the sample mean — mirrors `bootstrap.ts`."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class BootstrapResult:
    lo: float
    mean: float
    hi: float
    B: int


def bootstrap_mean_ci(
    values: np.ndarray | list[float],
    B: int = 1000,
    seed: int = 1337,
    alpha: float = 0.05,
) -> BootstrapResult:
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n == 0:
        return BootstrapResult(0.0, 0.0, 0.0, B)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    means = arr[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return BootstrapResult(lo=lo, mean=float(means.mean()), hi=hi, B=B)
