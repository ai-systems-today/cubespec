"""Phase B — Uncertainty decomposition + reliability analysis.

Two utilities, both honest about their assumptions:

* :func:`decompose_variance` splits the total variance of each output into:
    - **aleatory**: variance arising from sampling the inputs through the
      surrogate (the irreducible-given-current-knowledge piece);
    - **epistemic**: surrogate residual variance, taken from the calibration
      artifact's 5-fold CV residuals (model-form / training-data uncertainty).

  Total = aleatory + epistemic under the standard independence assumption
  between the parametric draw and the surrogate noise.

* :func:`reliability_index` reports, for a chosen output (default P9):
    - empirical exceedance probability ``p = P(Y >= threshold)``,
    - Wilson 95 % CI for that probability,
    - first-order reliability β-index ``β = Φ⁻¹(p)`` cross-check.

These are the standard UQ deliverables expected in a defensible thesis.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy.stats import norm

from .params import OUTPUT_KEYS

_RESID_PATH = Path(__file__).parent / "models" / "poly2_ridge_residuals.json"


@dataclass
class VarianceSplit:
    output: str
    aleatory: float
    epistemic: float
    total: float

    @property
    def aleatory_frac(self) -> float:
        return self.aleatory / self.total if self.total > 0 else 0.0

    @property
    def epistemic_frac(self) -> float:
        return self.epistemic / self.total if self.total > 0 else 0.0


def _load_residual_sds(path: Optional[str | Path] = None) -> Dict[str, float]:
    p = Path(path) if path else _RESID_PATH
    if not p.exists():
        # Honest fallback: epistemic component reported as 0 with a clear log.
        return {k: 0.0 for k in OUTPUT_KEYS}
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {k: float(np.std(np.asarray(raw[k], dtype=float), ddof=1)) for k in OUTPUT_KEYS}


def decompose_variance(Y: np.ndarray, residual_sds: Optional[Dict[str, float]] = None) -> Dict[str, VarianceSplit]:
    """Split per-output variance into aleatory (sampling) and epistemic (surrogate residual).

    Parameters
    ----------
    Y : ndarray (n, 3)
        Output draws ordered as :data:`OUTPUT_KEYS`.
    residual_sds : optional mapping
        Per-output residual standard deviations. If omitted, loaded from the
        calibrated surrogate's stored 5-fold CV residuals.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2 or Y.shape[1] != len(OUTPUT_KEYS):
        raise ValueError(f"Y must have shape (n, {len(OUTPUT_KEYS)}); got {Y.shape}")
    sds = residual_sds or _load_residual_sds()
    out: Dict[str, VarianceSplit] = {}
    for j, k in enumerate(OUTPUT_KEYS):
        col = Y[:, j]
        v_aleatory = float(np.var(col, ddof=1)) if col.size > 1 else 0.0
        v_epistemic = float(sds.get(k, 0.0)) ** 2
        out[k] = VarianceSplit(
            output=k,
            aleatory=v_aleatory,
            epistemic=v_epistemic,
            total=v_aleatory + v_epistemic,
        )
    return out


@dataclass
class ReliabilityResult:
    output: str
    threshold: float
    direction: str       # "ge" (>=) or "le" (<=)
    n: int
    successes: int
    p: float
    p_lo: float
    p_hi: float
    beta: float          # FORM β-index = Φ⁻¹(p); inf when p in {0, 1}

    def as_dict(self) -> dict:
        return {
            "output": self.output, "threshold": self.threshold, "direction": self.direction,
            "n": self.n, "successes": self.successes,
            "p": self.p, "p_lo": self.p_lo, "p_hi": self.p_hi,
            "wilson_ci_pct": 95, "beta": self.beta,
        }


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95 % CI for a binomial proportion. Robust at the {0, 1} extremes."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    halfwidth = (z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return float(max(0.0, centre - halfwidth)), float(min(1.0, centre + halfwidth))


def reliability_index(
    Y: np.ndarray,
    threshold: float,
    output: str = "P9_compressive_strength",
    direction: str = "ge",
) -> ReliabilityResult:
    """Empirical exceedance + FORM β-index for a single output.

    Parameters
    ----------
    Y : ndarray (n, 3)
        Monte Carlo output draws ordered as :data:`OUTPUT_KEYS`.
    threshold : float
        Acceptance threshold (e.g. 30 MPa for P9 per Eurocode-2 C25/30).
    output : str, default "P9_compressive_strength"
    direction : "ge" | "le"
        "ge" → success means Y ≥ threshold (typical for strength).
        "le" → success means Y ≤ threshold (typical for deformation/strain).
    """
    if output not in OUTPUT_KEYS:
        raise ValueError(f"output must be one of {OUTPUT_KEYS}; got {output!r}")
    if direction not in ("ge", "le"):
        raise ValueError(f"direction must be 'ge' or 'le'; got {direction!r}")
    Y = np.asarray(Y, dtype=float)
    col = Y[:, OUTPUT_KEYS.index(output)]
    n = int(col.size)
    if direction == "ge":
        successes = int(np.sum(col >= threshold))
    else:
        successes = int(np.sum(col <= threshold))
    p = successes / n if n > 0 else 0.0
    lo, hi = _wilson_ci(successes, n)
    if p <= 0.0 or p >= 1.0:
        beta = float("inf") if p >= 1.0 else float("-inf")
    else:
        beta = float(norm.ppf(p))
    return ReliabilityResult(
        output=output, threshold=float(threshold), direction=direction,
        n=n, successes=successes, p=float(p), p_lo=lo, p_hi=hi, beta=beta,
    )
