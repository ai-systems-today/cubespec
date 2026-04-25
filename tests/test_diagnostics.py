import numpy as np
from cubespec.diagnostics import rmse, mae, r2, bias, qq_pairs


def test_perfect_fit():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert rmse(y, y) == 0.0
    assert mae(y, y) == 0.0
    assert r2(y, y) == 1.0
    assert bias(y, y) == 0.0


def test_qq_shape():
    r = np.random.default_rng(0).normal(size=50)
    t, e = qq_pairs(r)
    assert t.shape == e.shape == (50,)
