"""Box-constrained optimisation of the calibrated surrogate.

Mirrors the TS implementation in ``src/components/dashboard/optimise.ts`` but
uses SciPy's L-BFGS-B (with multi-start) for production-grade quality. Both
implementations agree on the calibrated Poly2-ridge to within 1e-3 in Y on
golden parity fixtures (see python/tests/test_parity_calibrated.py).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Literal, Mapping, Tuple
import numpy as np
from scipy.optimize import minimize

from .params import CSP, PARAM_KEYS, OUTPUT_KEYS
from .surrogate import predict_calibrated

Direction = Literal["maximise", "minimise"]


@dataclass
class OptResult:
    x: Dict[str, float]
    outputs: Dict[str, float]
    value: float
    iterations: int
    starts: int
    converged: bool
    sensitivity: Dict[str, float]

    def as_dict(self) -> dict:
        return asdict(self)


def default_bounds(csp: CSP, k_sigma: float = 3.0) -> Dict[str, Tuple[float, float]]:
    return {k: (csp.params[k].mean - k_sigma * csp.params[k].sd,
                csp.params[k].mean + k_sigma * csp.params[k].sd) for k in PARAM_KEYS}


def _eval(x: np.ndarray, output_idx: int) -> float:
    Y = predict_calibrated(x.reshape(1, -1))
    return float(Y[0, output_idx])


def _objective(x: np.ndarray, output_idx: int, sign: float) -> float:
    return sign * _eval(x, output_idx)


def optimise(
    csp: CSP,
    output: str = "P9_compressive_strength",
    direction: Direction = "maximise",
    bounds: Mapping[str, Tuple[float, float]] | None = None,
    seed: int = 1337,
    n_starts: int = 8,
    maxiter: int = 200,
) -> OptResult:
    if output not in OUTPUT_KEYS:
        raise ValueError(f"output must be one of {OUTPUT_KEYS}")
    output_idx = OUTPUT_KEYS.index(output)
    sign = -1.0 if direction == "maximise" else 1.0

    b = bounds if bounds is not None else default_bounds(csp)
    lo = np.array([b[k][0] for k in PARAM_KEYS], dtype=float)
    hi = np.array([b[k][1] for k in PARAM_KEYS], dtype=float)
    bounds_seq = list(zip(lo, hi))

    rng = np.random.default_rng(seed)
    starts = [np.array([csp.params[k].mean for k in PARAM_KEYS], dtype=float)]
    for _ in range(max(0, n_starts - 1)):
        starts.append(lo + rng.random(len(PARAM_KEYS)) * (hi - lo))

    best = None
    total_iters = 0
    for x0 in starts:
        res = minimize(
            _objective, x0, args=(output_idx, sign),
            method="L-BFGS-B", bounds=bounds_seq,
            options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-8},
        )
        total_iters += int(res.nit)
        if best is None or res.fun < best.fun:
            best = res

    assert best is not None
    x_star = np.clip(best.x, lo, hi)
    Y = predict_calibrated(x_star.reshape(1, -1))[0]
    value = float(Y[output_idx])

    # Finite-difference sensitivities at the optimum (physical units).
    sens: Dict[str, float] = {}
    span = hi - lo
    for i, k in enumerate(PARAM_KEYS):
        h = max(1e-7, span[i] * 1e-5)
        xp = x_star.copy()
        xp[i] = min(hi[i], xp[i] + h)
        xm = x_star.copy()
        xm[i] = max(lo[i], xm[i] - h)
        yp = predict_calibrated(xp.reshape(1, -1))[0, output_idx]
        ym = predict_calibrated(xm.reshape(1, -1))[0, output_idx]
        dx = xp[i] - xm[i]
        sens[k] = float((yp - ym) / dx) if dx > 0 else 0.0

    return OptResult(
        x={k: float(x_star[i]) for i, k in enumerate(PARAM_KEYS)},
        outputs={k: float(Y[i]) for i, k in enumerate(OUTPUT_KEYS)},
        value=value,
        iterations=total_iters,
        starts=len(starts),
        converged=bool(best.success),
        sensitivity=sens,
    )
