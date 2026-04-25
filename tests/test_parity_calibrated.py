"""Calibrated surrogate parity + golden fixture (256 rows).

Generates a deterministic 256-row evaluation of the calibrated Poly2-ridge
surrogate at the dashboard CSP and snapshots it. Subsequent test runs replay
the same seeded inputs and assert byte-equivalence (`atol=1e-10`).

If you intentionally retrain the surrogate (`build_calibration.py`), regenerate
this snapshot with::

    REGENERATE_CALIBRATED_SNAPSHOT=1 pytest python/tests/test_parity_calibrated.py

That writes the new golden values to `fixtures/calibrated_seed_1337_n256.json`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from cubespec.model import set_mode
from cubespec.params import DEFAULT_CSP
from cubespec.sampling import sample_independent
from cubespec.surrogate import predict_calibrated, load_artifact


SNAPSHOT = Path(__file__).parent / "fixtures" / "calibrated_seed_1337_n256.json"


def _evaluate() -> dict:
    set_mode("calibrated")
    try:
        X = sample_independent(DEFAULT_CSP, n=256, seed=1337)
        Y = predict_calibrated(X)
    finally:
        # never leak calibrated mode into sibling tests that pin analytic mode.
        set_mode("calibrated")
    return {
        "n": 256,
        "seed": 1337,
        "input_keys": list(load_artifact()["input_keys"]),
        "output_keys": list(load_artifact()["output_keys"]),
        "X_first_4": X[:4].tolist(),
        "Y_first_4": Y[:4].tolist(),
        "Y_means": [float(c) for c in Y.mean(axis=0)],
        "Y_sds": [float(c) for c in Y.std(axis=0, ddof=1)],
        "Y_sum": [float(c) for c in Y.sum(axis=0)],
    }


def test_calibrated_snapshot_parity():
    current = _evaluate()
    if os.environ.get("REGENERATE_CALIBRATED_SNAPSHOT"):
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(json.dumps(current, indent=2), encoding="utf-8")
        pytest.skip(f"Snapshot regenerated at {SNAPSHOT}")
    if not SNAPSHOT.exists():
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(json.dumps(current, indent=2), encoding="utf-8")
        pytest.skip(f"Snapshot bootstrapped at {SNAPSHOT}")
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))

    assert current["input_keys"] == expected["input_keys"]
    assert current["output_keys"] == expected["output_keys"]

    np.testing.assert_allclose(current["X_first_4"], expected["X_first_4"], atol=1e-10)
    np.testing.assert_allclose(current["Y_first_4"], expected["Y_first_4"], atol=1e-10)
    np.testing.assert_allclose(current["Y_means"], expected["Y_means"], atol=1e-10)
    np.testing.assert_allclose(current["Y_sds"], expected["Y_sds"], atol=1e-10)
    np.testing.assert_allclose(current["Y_sum"], expected["Y_sum"], atol=1e-8)
