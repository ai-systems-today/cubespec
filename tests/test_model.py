import math
import pytest
from cubespec.model import compute_outputs, compute_outputs_batch, set_mode
from cubespec.params import DEFAULT_CSP, PARAM_KEYS
import numpy as np


@pytest.fixture(autouse=True)
def _analytic_mode():
    """These tests pin the analytic σ = F/A model. The calibrated surrogate
    has its own dedicated tests (see test_parity_calibrated.py)."""
    set_mode("analytic")
    yield
    set_mode("calibrated")


def _default_sample():
    return {k: DEFAULT_CSP.params[k].mean for k in PARAM_KEYS}


def test_default_p9_matches_dashboard():
    out = compute_outputs(_default_sample())
    # σ_axial = 1e6 / (149.9797 * 150.8522) ≈ 44.205 MPa
    # disturbance ≈ 0.0049 MPa, P9 ≈ 44.203 MPa
    assert abs(out["P9_compressive_strength"] - 44.20) < 0.05


def test_strain_via_hooke():
    out = compute_outputs(_default_sample())
    # E ≈ 0.255 * 2400^1.5 ≈ 29 988.7 MPa
    # ε ≈ σ_axial / E ≈ 44.205 / 29 988 ≈ 1.474e-3
    assert abs(out["P8_strain"] - 1.474e-3) < 5e-5


def test_batch_matches_scalar():
    s = _default_sample()
    X = np.array([[s[k] for k in PARAM_KEYS]])
    Y = compute_outputs_batch(X)
    out = compute_outputs(s)
    assert math.isclose(Y[0, 0], out["P7_def"], rel_tol=1e-12)
    assert math.isclose(Y[0, 1], out["P8_strain"], rel_tol=1e-12)
    assert math.isclose(Y[0, 2], out["P9_compressive_strength"], rel_tol=1e-12)
