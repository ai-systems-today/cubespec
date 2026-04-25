"""Phase F — Hypothesis property-based tests for the CubeSpec surrogate.

These tests exercise *invariants* the model must obey for any physically
meaningful input, regardless of the specific numeric draw — they complement
the example-based tests in ``test_model.py``.

Properties covered:

* **Monotonicity in F (axial load)** — increasing P4_Fx with all other
  inputs fixed must produce a non-decreasing P9 (compressive stress
  reading from the σ = F/A baseline minus a load-independent disturbance
  term).
* **Sign of P9** — for non-negative F and physically valid geometry,
  P9 ≥ 0 in the analytic regime (the disturbance term is bounded by
  half the axial term when |Fy|, |Fz| ≪ Fx).
* **Dimensional consistency for P8** — strain is dimensionless and must
  stay in a physically plausible band for the calibrated regime.
* **Scale invariance under unit conversion** — scaling all geometry by k
  scales area by k² and (with F scaled by k²) leaves P9 unchanged.
* **Determinism** — same inputs ⇒ same outputs (no hidden RNG).
* **Batch ↔ single-sample agreement** — vectorised path equals the
  per-sample dict path for every draw.
"""
from __future__ import annotations

import numpy as np
from hypothesis import given, settings, strategies as st, HealthCheck

from cubespec import compute_outputs, compute_outputs_batch
from cubespec.params import PARAM_KEYS


# ─── Strategies ──────────────────────────────────────────────────────────────
# Physically plausible bands around the default CSP (loose but non-crazy).

rho   = st.floats(min_value=2200.0,    max_value=2600.0,   allow_nan=False, allow_infinity=False)
dim   = st.floats(min_value=148.0,     max_value=152.0,    allow_nan=False, allow_infinity=False)
Fx    = st.floats(min_value=5.0e5,     max_value=2.0e6,    allow_nan=False, allow_infinity=False)
Fyz   = st.floats(min_value=-200.0,    max_value=200.0,    allow_nan=False, allow_infinity=False)


def _sample(rho_, dx, dy, dz, fx, fy, fz):
    return {
        "P0_rho": rho_,
        "P1_dx": dx,
        "P2_dy": dy,
        "P3_dz": dz,
        "P4_Fx": fx,
        "P5_Fy": fy,
        "P6_Fz": fz,
    }


# ─── Properties ──────────────────────────────────────────────────────────────

@settings(max_examples=80, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(rho_=rho, dx=dim, dy=dim, dz=dim, fx=Fx, fy=Fyz, fz=Fyz)
def test_monotonic_in_Fx_analytic(rho_, dx, dy, dz, fx, fy, fz):
    """Increasing P4_Fx must not decrease P9 (analytic mode)."""
    base = _sample(rho_, dx, dy, dz, fx, fy, fz)
    bumped = dict(base, P4_Fx=fx * 1.10)
    y0 = compute_outputs(base,    mode="analytic")["P9_compressive_strength"]
    y1 = compute_outputs(bumped, mode="analytic")["P9_compressive_strength"]
    assert y1 >= y0 - 1e-9, f"P9 decreased when Fx grew: {y0} -> {y1}"


@settings(max_examples=80, deadline=None)
@given(rho_=rho, dx=dim, dy=dim, dz=dim, fx=Fx, fy=Fyz, fz=Fyz)
def test_P9_nonnegative_when_disturbance_small(rho_, dx, dy, dz, fx, fy, fz):
    """Disturbance term is bounded; with |Fy|,|Fz| ≪ Fx, P9 must be ≥ 0."""
    s = _sample(rho_, dx, dy, dz, fx, fy, fz)
    out = compute_outputs(s, mode="analytic")
    A = dx * dy
    sigma_axial = fx / A
    sigma_dist = (abs(fy) + abs(fz)) / A
    if sigma_dist < 0.5 * sigma_axial:
        assert out["P9_compressive_strength"] > 0.0


@settings(max_examples=60, deadline=None)
@given(rho_=rho, dx=dim, dy=dim, dz=dim, fx=Fx, fy=Fyz, fz=Fyz)
def test_strain_in_physical_band(rho_, dx, dy, dz, fx, fy, fz):
    """ε = σ/E for normal-strength concrete must sit in [0, 5e-3]."""
    s = _sample(rho_, dx, dy, dz, fx, fy, fz)
    out = compute_outputs(s, mode="analytic")
    eps = out["P8_strain"]
    assert 0.0 <= eps <= 5e-3, f"strain out of physical band: {eps}"


@settings(max_examples=40, deadline=None)
@given(rho_=rho, dx=dim, dy=dim, dz=dim, fx=Fx)
def test_geometry_scale_invariance(rho_, dx, dy, dz, fx):
    """Scale geometry by k and Fx by k²: P9 must be invariant (analytic)."""
    k = 1.05
    base = _sample(rho_, dx, dy, dz, fx, 0.0, 0.0)
    scaled = _sample(rho_, dx * k, dy * k, dz, fx * k * k, 0.0, 0.0)
    p9_base = compute_outputs(base,   mode="analytic")["P9_compressive_strength"]
    p9_sc   = compute_outputs(scaled, mode="analytic")["P9_compressive_strength"]
    assert abs(p9_sc - p9_base) / max(abs(p9_base), 1e-9) < 1e-9


@settings(max_examples=40, deadline=None)
@given(rho_=rho, dx=dim, dy=dim, dz=dim, fx=Fx, fy=Fyz, fz=Fyz)
def test_determinism(rho_, dx, dy, dz, fx, fy, fz):
    s = _sample(rho_, dx, dy, dz, fx, fy, fz)
    a = compute_outputs(s, mode="analytic")
    b = compute_outputs(s, mode="analytic")
    assert a == b


@settings(max_examples=30, deadline=None)
@given(rho_=rho, dx=dim, dy=dim, dz=dim, fx=Fx, fy=Fyz, fz=Fyz)
def test_batch_matches_single(rho_, dx, dy, dz, fx, fy, fz):
    s = _sample(rho_, dx, dy, dz, fx, fy, fz)
    single = compute_outputs(s, mode="analytic")
    X = np.array([[s[k] for k in PARAM_KEYS]], dtype=float)
    batch = compute_outputs_batch(X, mode="analytic")[0]
    np.testing.assert_allclose(
        batch,
        [single["P7_def"], single["P8_strain"], single["P9_compressive_strength"]],
        rtol=1e-12, atol=1e-12,
    )
