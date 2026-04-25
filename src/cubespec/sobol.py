"""Sobol sensitivity indices (S1, ST) via the Saltelli A/B/AB scheme.

Wraps SALib for canonical numerics; the surrogate is evaluated through
`compute_outputs_batch` so the indices reflect the dashboard model exactly.
"""
from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd

from SALib.sample import saltelli
from SALib.analyze import sobol as salib_sobol

from .params import CSP, PARAM_KEYS, OUTPUT_KEYS
from .model import compute_outputs_batch


def _problem_from_csp(csp: CSP, span_sds: float = 3.0) -> dict:
    means = np.array(csp.means())
    sds = np.array(csp.sds())
    bounds = [[m - span_sds * s, m + span_sds * s] for m, s in zip(means, sds)]
    return {
        "num_vars": 7,
        "names": PARAM_KEYS,
        "bounds": bounds,
    }


def sobol_indices(
    csp: CSP,
    n_base: int = 1024,
    span_sds: float = 3.0,
    seed: int = 1337,
) -> pd.DataFrame:
    """Return a tidy DataFrame: factor, output, S1, ST, S1_conf, ST_conf."""
    problem = _problem_from_csp(csp, span_sds=span_sds)
    # Saltelli yields N*(2D+2) samples; D=7 ⇒ 16N rows.
    # SALib's saltelli sampler is deterministic given N (no seed kwarg in current versions).
    X = saltelli.sample(problem, n_base, calc_second_order=False)
    Y_all = compute_outputs_batch(X)
    rows = []
    for j, out in enumerate(OUTPUT_KEYS):
        Y = Y_all[:, j]
        Si = salib_sobol.analyze(
            problem, Y, calc_second_order=False, print_to_console=False, seed=seed
        )
        for i, k in enumerate(PARAM_KEYS):
            rows.append({
                "factor": k,
                "output": out,
                "S1": float(Si["S1"][i]),
                "ST": float(Si["ST"][i]),
                "S1_conf": float(Si["S1_conf"][i]),
                "ST_conf": float(Si["ST_conf"][i]),
            })
    return pd.DataFrame(rows)
