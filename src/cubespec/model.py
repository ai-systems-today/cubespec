"""Surrogate model f(X) — mirrors `src/components/dashboard/model.ts`.

Two modes (mirrors the dashboard):
  * "calibrated" (default) — Poly2-ridge surrogate loaded from
    ``cubespec/models/poly2_ridge.json`` (see ``python/scripts/build_calibration.py``).
  * "analytic"            — closed-form σ = F/A baseline.

Use :func:`set_mode` / :func:`get_mode` to switch globally, or pass
``mode=`` to the public helpers.
"""
from __future__ import annotations

from typing import Dict, Literal, Mapping
import numpy as np

Mode = Literal["calibrated", "analytic"]
_MODE: Mode = "calibrated"


def set_mode(mode: Mode) -> None:
    global _MODE
    if mode not in ("calibrated", "analytic"):
        raise ValueError(f"Unknown surrogate mode: {mode!r}")
    _MODE = mode


def get_mode() -> Mode:
    return _MODE


_K_E = 0.255  # tuned constant: 0.255 * 2400^1.5 ≈ 29 989 MPa


def youngs_modulus_mpa(rho_kgm3: float | np.ndarray) -> float | np.ndarray:
    return _K_E * np.power(rho_kgm3, 1.5)


def _analytic_dict(sample: Mapping[str, float]) -> Dict[str, float]:
    A_mm2 = sample["P1_dx"] * sample["P2_dy"]
    h_mm = sample["P3_dz"]
    sigma_axial = sample["P4_Fx"] / A_mm2
    sigma_disturb = (abs(sample["P5_Fy"]) + abs(sample["P6_Fz"])) / A_mm2
    P9 = sigma_axial - 0.5 * sigma_disturb
    E = float(youngs_modulus_mpa(sample["P0_rho"]))
    P8 = sigma_axial / E
    P7 = P8 * h_mm
    return {"P7_def": P7, "P8_strain": P8, "P9_compressive_strength": P9}


def _analytic_batch(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    rho, dx, dy, dz, Fx, Fy, Fz = (X[:, i] for i in range(7))
    A = dx * dy
    sigma_axial = Fx / A
    sigma_disturb = (np.abs(Fy) + np.abs(Fz)) / A
    P9 = sigma_axial - 0.5 * sigma_disturb
    E = youngs_modulus_mpa(rho)
    P8 = sigma_axial / E
    P7 = P8 * dz
    return np.column_stack([P7, P8, P9])


def compute_outputs(sample: Mapping[str, float], mode: Mode | None = None) -> Dict[str, float]:
    """Deterministic surrogate for a single input sample (dict of P0..P6)."""
    m = mode or _MODE
    if m == "analytic":
        return _analytic_dict(sample)
    # Lazy import to avoid scikit-learn dep at import time.
    from .surrogate import predict_calibrated_dict
    return predict_calibrated_dict(sample)


def compute_outputs_batch(X: np.ndarray, mode: Mode | None = None) -> np.ndarray:
    """Vectorised surrogate.

    Parameters
    ----------
    X : ndarray of shape (n, 7), columns ordered as PARAM_KEYS.
    mode : "calibrated" (default global) or "analytic".

    Returns
    -------
    ndarray of shape (n, 3) — columns P7_def, P8_strain, P9_compressive_strength.
    """
    m = mode or _MODE
    if m == "analytic":
        return _analytic_batch(X)
    from .surrogate import predict_calibrated
    return predict_calibrated(X)
