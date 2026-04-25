"""Residual diagnostics: RMSE, MAE, R², bias, Q-Q pairs."""
from __future__ import annotations

from typing import Tuple
import numpy as np


def _arrs(y_true, y_pred):
    return np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)


def rmse(y_true, y_pred) -> float:
    yt, yp = _arrs(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true, y_pred) -> float:
    yt, yp = _arrs(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def r2(y_true, y_pred) -> float:
    yt, yp = _arrs(y_true, y_pred)
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot == 0:
        return 0.0
    ss_res = float(np.sum((yt - yp) ** 2))
    return 1.0 - ss_res / ss_tot


def bias(y_true, y_pred) -> float:
    yt, yp = _arrs(y_true, y_pred)
    return float(np.mean(yp - yt))


def qq_pairs(residuals) -> Tuple[np.ndarray, np.ndarray]:
    """Return (theoretical_q, empirical_q) for a normal Q-Q plot."""
    from scipy.stats import norm

    r = np.asarray(residuals, dtype=float)
    r = np.sort(r)
    n = r.size
    if n == 0:
        return np.array([]), np.array([])
    p = (np.arange(1, n + 1) - 0.5) / n
    theoretical = norm.ppf(p)
    return theoretical, r
