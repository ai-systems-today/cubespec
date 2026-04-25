import math
import numpy as np
import pytest

from cubespec.uncertainty import decompose_variance, reliability_index, _wilson_ci
from cubespec.params import OUTPUT_KEYS


def _fake_outputs(n: int, mean: float, sd: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(size=(n, len(OUTPUT_KEYS)))
    return mean + sd * z


def test_decompose_variance_shapes_and_totals():
    Y = _fake_outputs(2_000, mean=44.0, sd=2.0)
    out = decompose_variance(Y, residual_sds={k: 0.5 for k in OUTPUT_KEYS})
    for k in OUTPUT_KEYS:
        s = out[k]
        assert s.output == k
        assert s.aleatory > 0
        assert math.isclose(s.epistemic, 0.25, rel_tol=1e-12)
        assert math.isclose(s.total, s.aleatory + s.epistemic, rel_tol=1e-12)
        assert 0.0 <= s.aleatory_frac <= 1.0
        assert 0.0 <= s.epistemic_frac <= 1.0
        assert math.isclose(s.aleatory_frac + s.epistemic_frac, 1.0, rel_tol=1e-12)


def test_decompose_variance_matches_sample_variance_with_no_epistemic():
    Y = _fake_outputs(5_000, mean=10.0, sd=3.0, seed=1)
    out = decompose_variance(Y, residual_sds={k: 0.0 for k in OUTPUT_KEYS})
    for j, k in enumerate(OUTPUT_KEYS):
        np.testing.assert_allclose(out[k].aleatory, np.var(Y[:, j], ddof=1), rtol=1e-12)
        assert out[k].epistemic == 0.0


def test_decompose_variance_loads_residuals_when_omitted():
    """Smoke: when no override is passed the artifact-stored residuals are used.
    We don't assert specific values (those drift with retraining) — only that
    the call succeeds and produces non-negative epistemic variance."""
    Y = _fake_outputs(500, mean=44.0, sd=2.0, seed=2)
    out = decompose_variance(Y)
    for k in OUTPUT_KEYS:
        assert out[k].epistemic >= 0.0


def test_decompose_variance_rejects_wrong_shape():
    with pytest.raises(ValueError):
        decompose_variance(np.zeros((10, 4)))


def test_wilson_ci_brackets_proportion():
    lo, hi = _wilson_ci(900, 1000)
    assert lo < 0.9 < hi
    assert 0.0 <= lo and hi <= 1.0


def test_wilson_ci_extremes():
    lo0, hi0 = _wilson_ci(0, 100)
    lo1, hi1 = _wilson_ci(100, 100)
    assert lo0 == 0.0 and 0.0 < hi0 < 0.1
    assert 0.9 < lo1 < 1.0 and hi1 == pytest.approx(1.0, abs=1e-9)


def test_reliability_index_strength_above_threshold():
    # Centre at 44 MPa, σ=2 MPa, threshold 40 MPa → P(Y≥40) ≈ 97.7%
    Y = _fake_outputs(20_000, mean=44.0, sd=2.0, seed=3)
    r = reliability_index(Y, threshold=40.0, output="P9_compressive_strength", direction="ge")
    assert r.n == 20_000
    assert 0.96 < r.p < 0.99
    assert r.p_lo < r.p < r.p_hi
    # β = Φ⁻¹(0.977) ≈ 2.0
    assert 1.7 < r.beta < 2.3


def test_reliability_index_deformation_below_threshold():
    Y = _fake_outputs(10_000, mean=0.21, sd=0.02, seed=4)
    r = reliability_index(Y, threshold=0.25, output="P7_def", direction="le")
    assert r.p > 0.95
    assert r.direction == "le"


def test_reliability_index_validates_inputs():
    Y = _fake_outputs(100, mean=1.0, sd=1.0)
    with pytest.raises(ValueError):
        reliability_index(Y, threshold=0.0, output="not_a_real_key")
    with pytest.raises(ValueError):
        reliability_index(Y, threshold=0.0, output="P9_compressive_strength", direction="lt")
