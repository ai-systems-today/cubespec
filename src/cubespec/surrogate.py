"""Calibrated Poly2-ridge surrogate loader (mirrors `src/components/dashboard/surrogate.ts`).

Loads the JSON artifact produced by `python/scripts/build_calibration.py`
and exposes ``predict(X)`` for batch evaluation.

Math is intentionally trivial so the TS port is byte-equivalent:
  z      = (x - mean) / scale          [per input dimension]
  feat_k = prod_j z_j ** powers[k][j]  [polynomial expansion]
  y      = intercept + Σ coef[k] * feat[k]
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Mapping
import numpy as np

from .params import PARAM_KEYS, OUTPUT_KEYS

_DEFAULT_PATH = Path(__file__).parent / "models" / "poly2_ridge.json"


@lru_cache(maxsize=4)
def load_artifact(path: str | None = None) -> dict:
    p = Path(path) if path else _DEFAULT_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Calibrated surrogate artifact not found at {p}. "
            "Run `python python/scripts/build_calibration.py` to build it."
        )
    return json.loads(p.read_text(encoding="utf-8"))


def _eval_one(model: dict, X: np.ndarray) -> np.ndarray:
    mean = np.asarray(model["scaler_mean"], dtype=float)
    scale = np.asarray(model["scaler_scale"], dtype=float)
    powers = np.asarray(model["powers"], dtype=int)        # (n_terms, 7)
    coef = np.asarray(model["coef"], dtype=float)          # (n_terms,)
    intercept = float(model["intercept"])
    Z = (X - mean) / scale                                 # (n, 7)
    # Polynomial features: Z**powers reduced over input axis (product).
    # Z[:, None, :] ** powers[None, :, :] -> (n, n_terms, 7), prod over last.
    feats = np.prod(Z[:, None, :] ** powers[None, :, :], axis=2)
    return intercept + feats @ coef


def predict_calibrated(X: np.ndarray, path: str | None = None) -> np.ndarray:
    """Evaluate the calibrated surrogate on a batch.

    Parameters
    ----------
    X : ndarray (n, 7) ordered as PARAM_KEYS.

    Returns
    -------
    ndarray (n, 3) with columns (P7_def, P8_strain, P9_compressive_strength).
    """
    art = load_artifact(path)
    X = np.asarray(X, dtype=float)
    cols = [_eval_one(art["models"][k], X) for k in OUTPUT_KEYS]
    return np.column_stack(cols)


def predict_calibrated_dict(sample: Mapping[str, float], path: str | None = None) -> Dict[str, float]:
    X = np.array([[sample[k] for k in PARAM_KEYS]], dtype=float)
    Y = predict_calibrated(X, path)[0]
    return {k: float(Y[i]) for i, k in enumerate(OUTPUT_KEYS)}
