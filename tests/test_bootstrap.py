import numpy as np
from cubespec.bootstrap import bootstrap_mean_ci


def test_ci_covers_true_mean_on_normal_data():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=10.0, scale=2.0, size=500)
    res = bootstrap_mean_ci(x, B=1000, seed=1337)
    assert res.lo < 10.0 < res.hi
    assert res.lo < res.mean < res.hi


def test_empty_input_safe():
    res = bootstrap_mean_ci([], B=200)
    assert res.mean == 0.0
