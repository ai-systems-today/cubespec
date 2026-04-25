"""Correlation matrices and Cholesky helpers — mirrors `correlation.ts`."""
from __future__ import annotations

import numpy as np


def identity_corr(n: int) -> np.ndarray:
    return np.eye(n)


def is_positive_definite(corr: np.ndarray) -> bool:
    try:
        np.linalg.cholesky(np.asarray(corr, dtype=float))
        return True
    except np.linalg.LinAlgError:
        return False


def cholesky(corr: np.ndarray) -> np.ndarray:
    return np.linalg.cholesky(np.asarray(corr, dtype=float))


def _named_preset(values: dict[tuple[int, int], float], n: int = 7) -> np.ndarray:
    M = np.eye(n)
    for (i, j), v in values.items():
        M[i, j] = v
        M[j, i] = v
    return M


# Geometry preset: dx, dy, dz mildly correlated (indices 1, 2, 3).
GEOMETRY_PRESET: np.ndarray = _named_preset({
    (1, 2): 0.35, (1, 3): 0.30, (2, 3): 0.40,
})

# Forces preset: Fx weakly anti-correlated with Fy/Fz lateral disturbance.
FORCES_PRESET: np.ndarray = _named_preset({
    (4, 5): -0.20, (4, 6): -0.15, (5, 6): 0.30,
})
