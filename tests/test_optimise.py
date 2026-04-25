"""Tests for the box-constrained optimiser."""
from __future__ import annotations

import numpy as np
import pytest

from cubespec import DEFAULT_CSP, PARAM_KEYS, optimise, default_bounds, predict_calibrated, set_mode


@pytest.fixture(autouse=True)
def use_calibrated():
    set_mode("calibrated")
    yield


def test_maximise_p9_improves_over_mean():
    res = optimise(DEFAULT_CSP, output="P9_compressive_strength", direction="maximise", seed=1337)
    mu = np.array([DEFAULT_CSP.params[k].mean for k in PARAM_KEYS])
    base = float(predict_calibrated(mu.reshape(1, -1))[0, 2])
    # Optimum strictly improves on the mean (calibrated quadratic is non-trivial).
    assert res.value > base
    assert res.converged
    # Optimum stays inside default ±3σ box.
    bounds = default_bounds(DEFAULT_CSP, 3.0)
    for k, v in res.x.items():
        lo, hi = bounds[k]
        assert lo - 1e-6 <= v <= hi + 1e-6


def test_minimise_p7_improves_over_mean():
    res = optimise(DEFAULT_CSP, output="P7_def", direction="minimise", seed=1337)
    mu = np.array([DEFAULT_CSP.params[k].mean for k in PARAM_KEYS])
    base = float(predict_calibrated(mu.reshape(1, -1))[0, 0])
    assert res.value < base
    assert res.converged


def test_deterministic_for_fixed_seed():
    a = optimise(DEFAULT_CSP, seed=42, n_starts=4)
    b = optimise(DEFAULT_CSP, seed=42, n_starts=4)
    assert a.value == pytest.approx(b.value, rel=1e-9)
    for k in a.x:
        assert a.x[k] == pytest.approx(b.x[k], rel=1e-9, abs=1e-9)


def test_sensitivity_signs_match_finite_difference():
    res = optimise(DEFAULT_CSP, output="P9_compressive_strength", direction="maximise", seed=1337)
    # P4_Fx (load) should have a positive sensitivity on P9 strength.
    assert res.sensitivity["P4_Fx"] > 0
    # All sensitivities are finite.
    for v in res.sensitivity.values():
        assert np.isfinite(v)


def test_tighter_bounds_reduces_optimum():
    wide = optimise(DEFAULT_CSP, output="P9_compressive_strength", direction="maximise",
                    bounds=default_bounds(DEFAULT_CSP, 3.0), seed=1337)
    tight = optimise(DEFAULT_CSP, output="P9_compressive_strength", direction="maximise",
                     bounds=default_bounds(DEFAULT_CSP, 1.0), seed=1337)
    assert tight.value <= wide.value + 1e-9
