"""Generate the figures used in docs/validation-report.md."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "reference_cubes.csv"
MODELS = ROOT / "src" / "cubespec" / "models"
OUT = ROOT.parent / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(ROOT / "src"))
from cubespec.surrogate import predict_calibrated  # noqa: E402

PARAM_KEYS = ["P0_rho", "P1_dx", "P2_dy", "P3_dz", "P4_Fx", "P5_Fy", "P6_Fz"]
OUTPUT_KEYS = ["P7_def", "P8_strain", "P9_compressive_strength"]
LABELS = {
    "P7_def": "P7 deformation (mm)",
    "P8_strain": "P8 strain (–)",
    "P9_compressive_strength": "P9 compressive strength (MPa)",
}

df = pd.read_csv(DATA)
X = df[PARAM_KEYS].to_numpy()
Y_meas = df[[f"{k}_meas" for k in OUTPUT_KEYS]].to_numpy()
Y_pred = predict_calibrated(X)
residuals = json.loads((MODELS / "poly2_ridge_residuals.json").read_text())

# ---------- Measured vs Predicted scatter ----------
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for i, k in enumerate(OUTPUT_KEYS):
    ax = axes[i]
    yt, yp = Y_meas[:, i], Y_pred[:, i]
    ax.scatter(yt, yp, s=10, alpha=0.45, color="#3b82f6", edgecolor="none")
    lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    # 95% PI band from residual SD
    s = float(np.std(np.array(residuals[k])))
    band = 1.96 * s
    ax.fill_between([lo, hi], [lo - band, hi - band], [lo + band, hi + band],
                    color="#3b82f6", alpha=0.10, label=f"95% PI (±{band:.3g})")
    ax.set_xlabel(f"Measured {LABELS[k]}")
    ax.set_ylabel(f"Predicted {LABELS[k]}")
    ax.set_title(k)
    ax.legend(loc="upper left", fontsize=8)
fig.suptitle("Calibrated Poly2-ridge surrogate · measured vs predicted (5-fold CV residual band)")
fig.tight_layout()
fig.savefig(OUT / "validation_scatter.png", dpi=140)
plt.close(fig)

# ---------- Residual Q-Q plot ----------
from scipy import stats
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for i, k in enumerate(OUTPUT_KEYS):
    ax = axes[i]
    r = np.array(residuals[k])
    stats.probplot(r, dist="norm", plot=ax)
    ax.set_title(f"Q-Q · {k}")
    ax.get_lines()[0].set_markerfacecolor("#3b82f6")
    ax.get_lines()[0].set_markeredgecolor("none")
    ax.get_lines()[1].set_color("black")
fig.suptitle("Residual normality (5-fold CV)")
fig.tight_layout()
fig.savefig(OUT / "validation_qq.png", dpi=140)
plt.close(fig)

# ---------- Convergence ribbon ----------
from cubespec.params import DEFAULT_CSP
from cubespec.sampling import sample_independent
ns = [2 ** k for k in range(6, 14)]  # 64..8192 — keep runtime modest
rng = np.random.default_rng(1337)
means = []
ci = []
for n in ns:
    X = sample_independent(DEFAULT_CSP, n=n, seed=1337)
    Y = predict_calibrated(X)[:, 2]
    m = float(Y.mean())
    s = float(Y.std(ddof=1))
    means.append(m)
    ci.append(1.96 * s / np.sqrt(n))
means = np.array(means); ci = np.array(ci)
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.fill_between(ns, means - ci, means + ci, color="#3b82f6", alpha=0.18, label="95% CI of mean")
ax.plot(ns, means, "o-", color="#1e40af", label="P9 mean (MPa)")
ax.set_xscale("log", base=2)
ax.set_xlabel("N (Monte Carlo runs)")
ax.set_ylabel("P9 estimate (MPa)")
ax.set_title("Convergence of P9 mean — calibrated surrogate")
ax.grid(True, which="both", alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "validation_convergence.png", dpi=140)
plt.close(fig)

print("Wrote figures to", OUT)
print("Final P9 mean estimate at N=8192:", means[-1], "+-", ci[-1])
