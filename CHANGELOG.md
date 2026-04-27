# Changelog

All notable changes to `cubespec` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]


## [0.1.0] — 2026-04-26

### Added
- First public release.
- 16-notebook curriculum under `notebooks/` mirroring every dashboard
  control 1:1 (Live MC, Sampling, Correlation, DOE full / fractional /
  multi-level / interactions / measurements, Sobol, RSM, Optimise,
  Uncertainty & Reliability, Calibrated vs Analytic, Full Report Parity).
- All 16 notebooks executed end-to-end via `nbmake` in CI
  (`.github/workflows/notebooks.yml`) — 17/17 passing locally.
- Colab + Binder + nbviewer + dashboard badges and `pip install`
  bootstrap cell auto-injected on every notebook by
  `scripts/normalise_notebooks.py` (idempotent).
- `cubespec.measurements` module: Madina-CSV ingestion with replicate
  averaging, schema auto-detection and `MeasurementSet.align()` for
  parity scatter / residual diagnostics. Re-exported from top-level
  `cubespec` package.
- `examples/madina_template.csv` blank template for lab measurements.
- `CITATION.cff` and `.zenodo.json` for thesis-grade citation.
- PyPI Trusted Publishing workflow (`.github/workflows/publish.yml`).
- Cross-language parity tests (`tests/test_parity.py`) locking the
  Python output to the TypeScript dashboard for `seed=1337`.
- Pre-rendered example plots under `examples/plots/output/`.
- `PUBLISHING.md` operator runbook for PyPI releases.
- Surrogate model `compute_outputs` / `compute_outputs_batch` mirroring
  the dashboard's `model.ts`.
- Independent, Latin Hypercube, and correlated multivariate-normal
  samplers.
- Full and fractional 2-level factorial DOE generators with main, 2-way,
  and 3-way interaction effects.
- Quadratic Response-Surface fit (`rsm.py`) with 2-D contour grid.
- Sobol S1 / ST sensitivity indices via the Saltelli scheme (`sobol.py`,
  wrapping `SALib`).
- Percentile bootstrap mean CI (`bootstrap.py`).
- Residual diagnostics: RMSE, MAE, R², bias, Q-Q pairs.
- Bit-for-bit `mulberry32` PRNG port (`rng.py`) for cross-language
  reproducibility with the React dashboard.
- `cubespec` command-line interface with `run`, `doe`, and `sobol`
  subcommands.
- `pyproject.toml` PEP 621 metadata for PyPI.

### Removed
- 4 superseded stub notebooks (`02_doe_full_vs_fractional`,
  `03_rsm_contour`, `04_sobol_sensitivity`, `05_bootstrap_ci`) replaced
  by the 16-notebook curriculum.

[Unreleased]: https://github.com/ai-systems-today/cubespec/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ai-systems-today/cubespec/releases/tag/v0.1.0
