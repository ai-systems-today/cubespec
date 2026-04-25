# Changelog

All notable changes to `cubespec` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Cross-language parity tests (`tests/test_parity.py`) locking the
  Python output to the TypeScript dashboard for `seed=1337`.
- Pre-rendered example plots under `examples/plots/output/`.
- `PUBLISHING.md` operator runbook for PyPI releases.

## [0.1.0] — 2026-04-22

### Added
- First public release.
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
- 5 Jupyter notebooks under `notebooks/` covering quickstart,
  full-vs-fractional DOE, RSM contours, Sobol sensitivity, and
  bootstrap confidence intervals.
- `pyproject.toml` PEP 621 metadata for PyPI.

[Unreleased]: https://github.com/ai-systems-today/cubespec/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ai-systems-today/cubespec/releases/tag/v0.1.0
