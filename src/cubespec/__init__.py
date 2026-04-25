"""CubeSpec — Monte Carlo, DOE, RSM and Sobol sensitivity for the 150 mm cube test.

Public API mirrors the TypeScript dashboard in `src/components/dashboard/`.
"""
from __future__ import annotations

from .params import (
    CSP,
    DEFAULT_CSP,
    PARAM_KEYS,
    OUTPUT_KEYS,
    load_csp_csv,
    save_csp_csv,
)
from .model import compute_outputs, compute_outputs_batch, set_mode, get_mode
from .surrogate import predict_calibrated, predict_calibrated_dict, load_artifact
from .rng import Mulberry32, randn
from .sampling import (
    sample_independent,
    sample_lhs,
    sample_mvn,
)
from .doe import (
    full_factorial,
    fractional_factorial,
    main_effects,
    interactions_2way,
    interactions_3way,
)
from .rsm import fit_quadratic, predict_grid
from .sobol import sobol_indices
from .bootstrap import bootstrap_mean_ci
from .diagnostics import rmse, mae, r2, bias, qq_pairs
from .correlation import (
    identity_corr,
    is_positive_definite,
    cholesky,
    GEOMETRY_PRESET,
    FORCES_PRESET,
)
from .uncertainty import decompose_variance, reliability_index, VarianceSplit, ReliabilityResult
from .optimise import optimise, default_bounds, OptResult
from . import measurements
from .measurements import (
    MeasurementRecord,
    MeasurementSet,
    parse_measurements,
    write_template,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "CSP",
    "DEFAULT_CSP",
    "PARAM_KEYS",
    "OUTPUT_KEYS",
    "load_csp_csv",
    "save_csp_csv",
    "compute_outputs",
    "compute_outputs_batch",
    "set_mode",
    "get_mode",
    "predict_calibrated",
    "predict_calibrated_dict",
    "load_artifact",
    "Mulberry32",
    "randn",
    "sample_independent",
    "sample_lhs",
    "sample_mvn",
    "full_factorial",
    "fractional_factorial",
    "main_effects",
    "interactions_2way",
    "interactions_3way",
    "fit_quadratic",
    "predict_grid",
    "sobol_indices",
    "bootstrap_mean_ci",
    "rmse",
    "mae",
    "r2",
    "bias",
    "qq_pairs",
    "identity_corr",
    "is_positive_definite",
    "cholesky",
    "GEOMETRY_PRESET",
    "FORCES_PRESET",
    "decompose_variance",
    "reliability_index",
    "VarianceSplit",
    "ReliabilityResult",
    "optimise",
    "default_bounds",
    "OptResult",
    "measurements",
    "MeasurementRecord",
    "MeasurementSet",
    "parse_measurements",
    "write_template",
]
