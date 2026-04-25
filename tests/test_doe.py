import pytest

from cubespec.doe import full_factorial, fractional_factorial, main_effects
from cubespec.model import set_mode
from cubespec.params import DEFAULT_CSP


@pytest.fixture
def model_mode(request):
    """Parametrised surrogate-mode fixture: switches mode for the test
    body, then restores the calibrated default afterwards."""
    set_mode(request.param)
    yield request.param
    set_mode("calibrated")


@pytest.mark.parametrize("model_mode", ["analytic", "calibrated"], indirect=True)
def test_full_factorial_2_level_run_count(model_mode):
    df = full_factorial(DEFAULT_CSP, levels=2)
    assert len(df) == 2 ** 7  # 128


@pytest.mark.parametrize("model_mode", ["analytic", "calibrated"], indirect=True)
def test_fractional_half_run_count(model_mode):
    df = fractional_factorial(DEFAULT_CSP, fraction="1/2")
    assert len(df) == 64


@pytest.mark.parametrize("model_mode", ["analytic", "calibrated"], indirect=True)
def test_fractional_quarter_run_count(model_mode):
    df = fractional_factorial(DEFAULT_CSP, fraction="1/4")
    assert len(df) == 32


@pytest.mark.parametrize("model_mode", ["analytic"], indirect=True)
def test_main_effects_p4_dominates_p9_analytic(model_mode):
    """The analytic σ = F/A model implies P4 (axial load) dominates P9.
    The calibrated surrogate may rank factors differently within its
    narrower CSP envelope (see docs/validation-report.md)."""
    df = full_factorial(DEFAULT_CSP, levels=2)
    eff = main_effects(df)
    p9 = eff[eff["output"] == "P9_compressive_strength"].set_index("factor")["abs"]
    assert p9["P4_Fx"] > p9["P0_rho"]
    assert p9["P4_Fx"] > p9["P5_Fy"]


@pytest.mark.parametrize("model_mode", ["calibrated"], indirect=True)
def test_main_effects_calibrated_runs(model_mode):
    """Calibrated surrogate: structural smoke test — main_effects produces
    a non-empty table with finite values for every factor × output cell."""
    import numpy as np
    df = full_factorial(DEFAULT_CSP, levels=2)
    eff = main_effects(df)
    assert len(eff) > 0
    assert np.isfinite(eff["abs"]).all()
    assert (eff["abs"] >= 0).all()
