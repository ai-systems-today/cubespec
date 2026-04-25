"""Response Surface Methodology — quadratic OLS fit and 2-D contour grid."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np



@dataclass
class QuadraticModel:
    """Quadratic surrogate y = β₀ + Σβᵢxᵢ + Σβᵢⱼxᵢxⱼ + Σβᵢᵢxᵢ²."""
    coef: np.ndarray
    feature_names: List[str]
    r2: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        Phi = _quadratic_features(np.asarray(X, dtype=float))
        return Phi @ self.coef


def _quadratic_features(X: np.ndarray) -> np.ndarray:
    n, p = X.shape
    cols = [np.ones(n)]
    names = ["const"]
    for i in range(p):
        cols.append(X[:, i])
        names.append(f"x{i}")
    for i in range(p):
        for j in range(i, p):
            cols.append(X[:, i] * X[:, j])
            names.append(f"x{i}x{j}")
    return np.column_stack(cols)


def fit_quadratic(X: np.ndarray, y: np.ndarray) -> QuadraticModel:
    """Ordinary least-squares quadratic fit. X shape (n, p), y shape (n,)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    Phi = _quadratic_features(X)
    coef, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    yhat = Phi @ coef
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    n, p = X.shape
    feature_names = ["const"] + [f"x{i}" for i in range(p)]
    for i in range(p):
        for j in range(i, p):
            feature_names.append(f"x{i}x{j}")
    return QuadraticModel(coef=coef, feature_names=feature_names, r2=r2)


def predict_grid(
    model: QuadraticModel,
    base: np.ndarray,
    factor_a: int,
    factor_b: int,
    span_a: tuple[float, float],
    span_b: tuple[float, float],
    grid: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep two factors over a grid, holding others at `base`. Returns (A, B, Z)."""
    a = np.linspace(span_a[0], span_a[1], grid)
    b = np.linspace(span_b[0], span_b[1], grid)
    A, B = np.meshgrid(a, b)
    n = grid * grid
    X = np.tile(base, (n, 1))
    X[:, factor_a] = A.ravel()
    X[:, factor_b] = B.ravel()
    Z = model.predict(X).reshape(grid, grid)
    return A, B, Z
