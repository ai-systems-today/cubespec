# cubespec

> **Monte-Carlo, DOE, RSM and Sobol sensitivity for the 150 mm concrete
> cube compressive test** — Python companion to the
> [Sensitive-Spark dashboard](https://sensitive-spark.lovable.app), with
> a bit-for-bit parity contract against the TypeScript implementation.

[![CI](https://github.com/ai-systems-today/cubespec/actions/workflows/ci.yml/badge.svg)](https://github.com/ai-systems-today/cubespec/actions/workflows/ci.yml)
[![Notebooks](https://github.com/ai-systems-today/cubespec/actions/workflows/notebooks.yml/badge.svg)](https://github.com/ai-systems-today/cubespec/actions/workflows/notebooks.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Live demo](https://img.shields.io/badge/demo-sensitive--spark.lovable.app-blue)](https://sensitive-spark.lovable.app)
<!-- Enable after first PyPI release: -->
<!-- [![PyPI](https://img.shields.io/pypi/v/cubespec.svg)](https://pypi.org/project/cubespec/) -->
<!-- [![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://pypi.org/project/cubespec/) -->
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.PLACEHOLDER.svg)](https://doi.org/10.5281/zenodo.PLACEHOLDER) -->

Two front ends, one set of equations:

- 🌐 **Web dashboard** — React + TypeScript, runs in the browser, no install: <https://sensitive-spark.lovable.app>
- 🐍 **Python package** — `cubespec`, this repo. Notebook-, CLI- and library-friendly.

Both implementations share the same surrogate model and PRNG
(`mulberry32`), so the same seed gives the same numbers in either
environment, validated by `tests/test_parity.py` at `1e-9` tolerance.

---

## What it does

Given the **Cube Specification Parameters** (CSP) — material density,
geometry (`dx, dy, dz`) and forces (`Fx, Fy, Fz`) — `cubespec` computes
the three dependent outputs P7 (deformation), P8 (strain), and **P9
(compressive strength)** through a calibrated surrogate, and quantifies
how uncertainty propagates through it.

| Capability | Web dashboard | Python (`cubespec`) |
|---|:---:|:---:|
| Live Monte-Carlo with convergence tracking | ✅ | ✅ |
| Independent / LHS / correlated MVN sampling | ✅ | ✅ |
| Full & fractional 2^(k-p) factorial DOE (1/2, 1/4, 1/8) | ✅ | ✅ |
| Main, 2-way & 3-way interaction effects | ✅ | ✅ |
| Quadratic Response-Surface fit + 2-D contours | ✅ | ✅ |
| Sobol S1 / ST sensitivity (Saltelli scheme) | ✅ | ✅ |
| Percentile bootstrap CI (B = 1000) | ✅ | ✅ |
| Residual diagnostics (RMSE, MAE, R², Q-Q) | ✅ | ✅ |
| CSV / JSON export | ✅ | ✅ |
| CLI | – | ✅ |
| Calibrated surrogate (poly-2 ridge + residuals) | ✅ | ✅ |

---

## Install

Until the first PyPI release, install directly from GitHub:

```bash
pip install "git+https://github.com/ai-systems-today/cubespec.git"
```

Or from a local clone (for development):

```bash
git clone https://github.com/ai-systems-today/cubespec.git
cd cubespec
pip install -e ".[plot,cli,dev]"
```

Once on PyPI:

```bash
# pip install cubespec
```

---

## Quickstart — Python API

```python
from cubespec import (
    DEFAULT_CSP, sample_independent, compute_outputs_batch,
    bootstrap_mean_ci, sobol_indices,
)

# 50 000-draw Monte-Carlo run on the default CSP
X = sample_independent(DEFAULT_CSP, n=50_000, seed=1337)
Y = compute_outputs_batch(X)               # columns: P7, P8, P9
print(f"P9 mean: {Y[:, 2].mean():.2f} MPa")
# → P9 mean: 44.20 MPa  (matches the dashboard at the same seed)

# 95 % bootstrap CI on the P9 mean
lo, hi = bootstrap_mean_ci(Y[:, 2], B=1000, seed=1337, alpha=0.05)
print(f"95% CI: [{lo:.2f}, {hi:.2f}] MPa")

# Sobol indices (Saltelli, base N = 1024)
print(sobol_indices(DEFAULT_CSP, n_base=1024))
```

## Quickstart — CLI

```bash
# 50k Monte-Carlo run + report JSON
cubespec run --csp examples/default_csp.csv --n 50000 --output report.json

# Fractional 2^(7-3) factorial design
cubespec doe --csp examples/default_csp.csv --design fractional-1/8 --output design.csv

# Sobol sensitivity (Saltelli, base N = 1024)
cubespec sobol --csp examples/default_csp.csv --n 1024 --output sobol.csv

# Re-run a saved bundle from the dashboard's Reproduce button
cubespec replay run.json
```

---

## Methods (1-page summary)

The surrogate is

```
σ = F / A          A = dx · dy             (engineering stress, MPa)
E ≈ 0.255 · ρ¹·⁵                            (concrete stiffness law)
ε = σ / E                                   (Hooke)
δ = ε · h          h = dz                   (axial deformation)
```

with the disturbance term `−0.5·(|Fy|+|Fz|)/A` subtracted from σ to give P9.

**Sampling.** Independent Gaussian, Latin Hypercube (Owen-shuffled), or
multivariate normal with a Cholesky factor of an editable correlation matrix.

**DOE.** Two-level full factorial (2⁷ = 128 runs) and fractional designs
2^(7-1) / 2^(7-2) / 2^(7-3) generated via pyDOE2 (Resolution VII / IV /
IV). Effects estimated as the contrast of mean responses at coded ±1
levels.

**RSM.** Quadratic OLS on `[1, xᵢ, xᵢxⱼ, xᵢ²]`; 2-D contour grids over
any factor pair, others held at the run mean.

**Sobol.** Saltelli A/B/AB matrices; S1 and ST estimated via SALib. The
default span is ±3σ around each parameter mean.

**Confidence.** Percentile bootstrap (B = 1000, seed 1337) on the P9 mean.

Full theory and references in the bilingual thesis at
<https://sensitive-spark.lovable.app/thesis>.

---

## Notebooks

Seventeen Jupyter notebooks in [`notebooks/`](notebooks/) walk through
the full curriculum end-to-end:

| # | Topic |
|---:|---|
| 01 | Quickstart |
| 02 | CSP I/O |
| 03 | Live controls |
| 04 | Sampling strategies |
| 05 | Correlation presets |
| 06 | Full factorial DOE |
| 07 | Fractional DOE |
| 08 | Multilevel DOE |
| 09 | DOE interactions |
| 10 | DOE with measurements |
| 11 | Sobol indices |
| 12 | Response Surface Methodology |
| 13 | Optimisation |
| 14 | Uncertainty & reliability |
| 15 | Calibrated vs analytic surrogate |
| 16 | Full report parity |

All seventeen execute end-to-end in CI on every push to `main`
(see [`notebooks.yml`](.github/workflows/notebooks.yml)).

---

## Reproducibility

The thesis bundle is reproducible three ways:

1. **In-app Reproduce button** — open the dashboard at
   <https://sensitive-spark.lovable.app>, go to the Report tab, click
   **Reproduce thesis bundle**. Downloads a ZIP with every figure (PNG)
   and table (CSV) plus the commit SHA.
2. **Python CLI** — `cubespec replay run.json` re-executes a saved
   dashboard run.
3. **Parity contract** — `pytest tests/test_parity.py` matches the
   TypeScript fixtures bit-for-bit at `1e-9` tolerance.

---

## Citation

Use [`CITATION.cff`](CITATION.cff) — GitHub renders a *Cite this
repository* button automatically. A Zenodo DOI will be minted with the
first PyPI release.

---

## License

MIT — see [`LICENSE`](LICENSE).
