"""Design of Experiments — full and fractional factorials, main + interaction effects.

Uses pyDOE2 for design generation when available, with a NumPy fallback
for plain full factorials.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple
from itertools import combinations, product
import numpy as np
import pandas as pd

from .params import CSP, PARAM_KEYS, OUTPUT_KEYS
from .model import compute_outputs_batch


def _coded_to_real(coded: np.ndarray, csp: CSP, span_sds: float = 1.0) -> np.ndarray:
    """Map ±1 coded levels to real values using mean ± span_sds·sd."""
    means = np.array(csp.means())
    sds = np.array(csp.sds())
    return means + coded * sds * span_sds


def full_factorial(csp: CSP, levels: int = 2, span_sds: float = 1.0) -> pd.DataFrame:
    """Full factorial across all 7 parameters at `levels` evenly-spaced settings.

    Returns a DataFrame with the 7 input columns (real units) plus the
    surrogate outputs P7/P8/P9.
    """
    if levels < 2:
        raise ValueError("levels must be >= 2")
    grid = np.linspace(-1.0, 1.0, levels)
    coded = np.array(list(product(grid, repeat=7)))
    real = _coded_to_real(coded, csp, span_sds)
    Y = compute_outputs_batch(real)
    df = pd.DataFrame(real, columns=PARAM_KEYS)
    for i, k in enumerate(OUTPUT_KEYS):
        df[k] = Y[:, i]
    # Keep coded levels too for effect computation.
    for i, k in enumerate(PARAM_KEYS):
        df[f"{k}_coded"] = coded[:, i]
    return df


def _fracfact(generator: str) -> np.ndarray:
    """Build a 2-level fractional factorial design from a generator string.

    Tokens are letters a..z for base factors and combinations like 'abc'
    for derived (aliased) columns. Returns a matrix of ±1 entries.
    Pure-Python replacement for pyDOE2.fracfact (which is broken on Py3.13).
    """
    tokens = generator.lower().split()
    base_letters = sorted({c for tok in tokens for c in tok})
    n_base = len(base_letters)
    base_idx = {c: i for i, c in enumerate(base_letters)}
    # Full 2^n_base on the base columns.
    base = np.array(list(product([-1, 1], repeat=n_base)))
    cols = []
    for tok in tokens:
        col = np.ones(base.shape[0], dtype=int)
        for c in tok:
            col = col * base[:, base_idx[c]]
        cols.append(col)
    return np.column_stack(cols).astype(float)


def fractional_factorial(
    csp: CSP,
    fraction: str = "1/2",
    span_sds: float = 1.0,
) -> pd.DataFrame:
    """Two-level fractional factorial 2^(7-p).

    fraction: '1/2' -> 2^(7-1) = 64 runs (Resolution VII)
              '1/4' -> 2^(7-2) = 32 runs (Resolution IV)
              '1/8' -> 2^(7-3) = 16 runs (Resolution IV)
    """
    designs = {
        "1/2": "a b c d e f abcdef",
        "1/4": "a b c d e abcd abce",
        "1/8": "a b c d abc abd acd",
    }
    if fraction not in designs:
        raise ValueError(f"Unknown fraction {fraction!r}; pick one of {list(designs)}")
    coded = _fracfact(designs[fraction])
    real = _coded_to_real(coded, csp, span_sds)
    Y = compute_outputs_batch(real)
    df = pd.DataFrame(real, columns=PARAM_KEYS)
    for i, k in enumerate(OUTPUT_KEYS):
        df[k] = Y[:, i]
    for i, k in enumerate(PARAM_KEYS):
        df[f"{k}_coded"] = coded[:, i]
    return df


def _coded_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[[f"{k}_coded" for k in PARAM_KEYS]].to_numpy()


def main_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate main effects (mean response at +1 minus mean response at -1)."""
    coded = _coded_matrix(df)
    rows = []
    for j, k in enumerate(PARAM_KEYS):
        plus = coded[:, j] > 0
        minus = ~plus
        for out in OUTPUT_KEYS:
            y = df[out].to_numpy()
            eff = y[plus].mean() - y[minus].mean()
            rows.append({"factor": k, "output": out, "effect": eff, "abs": abs(eff)})
    return pd.DataFrame(rows)


def _interaction_effect(coded: np.ndarray, y: np.ndarray, idxs: Iterable[int]) -> float:
    sign = np.ones(len(y))
    for j in idxs:
        sign = sign * coded[:, j]
    return float((y * sign).mean())


def interactions_2way(df: pd.DataFrame) -> pd.DataFrame:
    coded = _coded_matrix(df)
    rows = []
    for i, j in combinations(range(7), 2):
        for out in OUTPUT_KEYS:
            y = df[out].to_numpy()
            eff = _interaction_effect(coded, y, (i, j))
            rows.append({
                "factors": f"{PARAM_KEYS[i]}*{PARAM_KEYS[j]}",
                "output": out,
                "effect": eff,
                "abs": abs(eff),
            })
    return pd.DataFrame(rows)


def interactions_3way(df: pd.DataFrame) -> pd.DataFrame:
    coded = _coded_matrix(df)
    rows = []
    for i, j, k in combinations(range(7), 3):
        for out in OUTPUT_KEYS:
            y = df[out].to_numpy()
            eff = _interaction_effect(coded, y, (i, j, k))
            rows.append({
                "factors": f"{PARAM_KEYS[i]}*{PARAM_KEYS[j]}*{PARAM_KEYS[k]}",
                "output": out,
                "effect": eff,
                "abs": abs(eff),
            })
    return pd.DataFrame(rows)
