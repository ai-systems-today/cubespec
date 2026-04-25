import pytest
from cubespec.sobol import sobol_indices
from cubespec.model import set_mode
from cubespec.params import DEFAULT_CSP


@pytest.fixture
def model_mode(request):
    set_mode(request.param)
    yield request.param
    set_mode("calibrated")


@pytest.mark.parametrize("model_mode", ["analytic"], indirect=True)
def test_sobol_p4_dominates_p9_analytic(model_mode):
    df = sobol_indices(DEFAULT_CSP, n_base=256, seed=42)
    p9 = df[df["output"] == "P9_compressive_strength"].set_index("factor")
    # P4 (axial load) should dominate the variance of P9.
    assert p9.loc["P4_Fx", "ST"] > 0.5
    # All ST should be in [0, ~1.2] (small overshoot allowed for finite-sample noise).
    assert (df["ST"] >= -0.05).all()
    assert (df["ST"] <= 1.2).all()


@pytest.mark.parametrize("model_mode", ["calibrated"], indirect=True)
def test_sobol_calibrated_smoke(model_mode):
    """Calibrated surrogate: structural sanity — Sobol returns a finite,
    well-formed table for every (factor, output) cell."""
    import numpy as np
    df = sobol_indices(DEFAULT_CSP, n_base=128, seed=42)
    assert len(df) > 0
    assert np.isfinite(df["S1"]).all()
    assert np.isfinite(df["ST"]).all()
    # ST must be at least as large as S1 (up to MC noise).
    assert (df["ST"] - df["S1"] >= -0.15).all()
