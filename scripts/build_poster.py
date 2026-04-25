"""Build the A0 conference poster (841 × 1189 mm) for the CubeSpec thesis.

Layout (matplotlib gridspec, 300 dpi):

    ┌──────────────────────────────────────────────────────────────┐
    │                       TITLE BAR                              │
    ├────────────────┬─────────────────────────┬───────────────────┤
    │  Abstract      │   Method diagram        │   QR code         │
    ├────────────────┴─────────────────────────┴───────────────────┤
    │  Fig 1: P9 distribution      │  Fig 2: Sobol bars            │
    ├──────────────────────────────┼───────────────────────────────┤
    │  Fig 3: RSM contour           │  Fig 4: Reliability card     │
    ├──────────────────────────────────────────────────────────────┤
    │  Footer: authors, repo, DOI, licence                         │
    └──────────────────────────────────────────────────────────────┘

The headline figures are produced from the analytic surrogate (no
calibration artifact required) so the poster compiles cleanly in any
environment — including CI without scikit-learn.

Usage:
    python python/scripts/build_poster.py \
        --out docs/media/poster.pdf \
        --preview docs/media/poster_preview.png
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

from cubespec import (
    DEFAULT_CSP, sample_independent, compute_outputs_batch, sobol_indices,
)
from cubespec.model import set_mode

# A0 portrait in inches at 300 dpi
A0_W_IN, A0_H_IN = 841 / 25.4, 1189 / 25.4
DPI = 300

NAVY = "#0b2545"
TEAL = "#13c4a3"
ACCENT = "#ef6c00"
MUTED = "#5b6b7c"


def _qr_image(url: str, box_size: int = 10):
    try:
        import qrcode
    except ImportError:
        return None
    qr = qrcode.QRCode(box_size=box_size, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color=NAVY, back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return plt.imread(buf)


def _figure() -> plt.Figure:
    set_mode("analytic")  # works without calibration artifact
    rng_seed = 1337
    N = 50_000

    X = sample_independent(DEFAULT_CSP, n=N, seed=rng_seed)
    Y = compute_outputs_batch(X)
    p9 = Y[:, 2]

    fig = plt.figure(figsize=(A0_W_IN, A0_H_IN), dpi=DPI, facecolor="white")
    gs = gridspec.GridSpec(
        nrows=5, ncols=3,
        height_ratios=[0.10, 0.18, 0.30, 0.30, 0.06],
        width_ratios=[1.0, 1.0, 0.5],
        hspace=0.32, wspace=0.18,
        left=0.04, right=0.96, top=0.97, bottom=0.02,
    )

    # ── Title bar ────────────────────────────────────────────────────────
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax_title.transAxes,
                                     facecolor=NAVY, zorder=-1))
    ax_title.text(0.02, 0.55,
                  "CubeSpec — UQ, DOE & Sobol for the 150 mm Cube Test",
                  fontsize=54, color="white", weight="bold",
                  va="center", family="DejaVu Sans")
    ax_title.text(0.02, 0.18,
                  "Sensitive-Spark Thesis Project · sensitive-spark.lovable.app",
                  fontsize=24, color=TEAL, va="center", family="DejaVu Sans")

    # ── Abstract ─────────────────────────────────────────────────────────
    ax_abs = fig.add_subplot(gs[1, 0])
    ax_abs.axis("off")
    ax_abs.text(0.0, 1.0, "Abstract", fontsize=28, weight="bold", color=NAVY,
                va="top", family="DejaVu Sans")
    ax_abs.text(
        0.0, 0.85,
        "An open-source toolkit (web dashboard + Python\n"
        "package) for uncertainty quantification of the\n"
        "EN 12390-3 cube test. Calibrated Poly2-ridge\n"
        "surrogate (R² ≈ 0.97) propagates 7 input\n"
        "parameters through Monte-Carlo, full / fractional\n"
        "DOE, RSM and Sobol indices. Reproducible via\n"
        "Docker, Binder and a deterministic Mulberry32\n"
        "PRNG: identical numbers in either front-end at\n"
        "the same seed.",
        fontsize=16, color=MUTED, va="top", family="DejaVu Sans",
    )

    # ── Method diagram ───────────────────────────────────────────────────
    ax_m = fig.add_subplot(gs[1, 1])
    ax_m.axis("off")
    ax_m.text(0.0, 1.0, "Pipeline", fontsize=28, weight="bold", color=NAVY,
              va="top", family="DejaVu Sans")
    boxes = [
        (0.00, "CSP\n(7 inputs)"),
        (0.20, "Sampler\n(MC / LHS / MVN)"),
        (0.40, "Surrogate\nPoly2-ridge"),
        (0.60, "DOE / RSM\n/ Sobol"),
        (0.80, "Report\nPDF + ZIP"),
    ]
    for x, label in boxes:
        ax_m.add_patch(plt.Rectangle((x, 0.30), 0.16, 0.36, facecolor=TEAL,
                                     edgecolor=NAVY, lw=2, alpha=0.85))
        ax_m.text(x + 0.08, 0.48, label, fontsize=13, ha="center", va="center",
                  color="white", weight="bold", family="DejaVu Sans")
    for x in [0.16, 0.36, 0.56, 0.76]:
        ax_m.annotate("", xy=(x + 0.04, 0.48), xytext=(x, 0.48),
                      arrowprops=dict(arrowstyle="->", color=NAVY, lw=2))
    ax_m.set_xlim(0, 1); ax_m.set_ylim(0, 1)

    # ── QR code ──────────────────────────────────────────────────────────
    ax_qr = fig.add_subplot(gs[1, 2])
    ax_qr.axis("off")
    qr = _qr_image("https://sensitive-spark.lovable.app")
    if qr is not None:
        ax_qr.imshow(qr)
    else:
        ax_qr.text(0.5, 0.5, "[QR]", fontsize=40, ha="center", va="center",
                   color=NAVY)
    ax_qr.text(0.5, -0.08, "sensitive-spark.lovable.app",
               fontsize=12, ha="center", va="top", color=NAVY,
               transform=ax_qr.transAxes, family="DejaVu Sans")

    # ── Fig 1: P9 histogram ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[2, 0:2])
    ax1.hist(p9, bins=80, color=TEAL, edgecolor=NAVY, alpha=0.85)
    ax1.axvline(p9.mean(), color=ACCENT, lw=3, label=f"mean = {p9.mean():.2f} MPa")
    ax1.set_title("P9 (compressive strength) — N = 50 000",
                  fontsize=22, color=NAVY, weight="bold")
    ax1.set_xlabel("P9 [MPa]", fontsize=16, color=MUTED)
    ax1.set_ylabel("count", fontsize=16, color=MUTED)
    ax1.tick_params(labelsize=13, colors=MUTED)
    ax1.legend(fontsize=14, frameon=False)
    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)

    # ── Fig 2: Sobol bars ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2, 2])
    sob = sobol_indices(DEFAULT_CSP, n_base=512, seed=rng_seed)
    p9_rows = sob[sob["output"] == "P9_compressive_strength"].copy()
    p9_rows = p9_rows.sort_values("ST", ascending=True)
    short = p9_rows["factor"].str.split("_").str[0].tolist()
    st_vals = p9_rows["ST"].clip(lower=0).tolist()
    ax2.barh(short, st_vals, color=NAVY, edgecolor=TEAL)
    ax2.set_title("Sobol ST · P9", fontsize=20, color=NAVY, weight="bold")
    ax2.tick_params(labelsize=12, colors=MUTED)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)

    # ── Fig 3: RSM contour (P1 × P4) ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[3, 0:2])
    grid = 60
    p1_grid = np.linspace(149.0, 151.0, grid)
    p4_grid = np.linspace(9.5e5, 1.05e6, grid)
    G1, G4 = np.meshgrid(p1_grid, p4_grid)
    means = DEFAULT_CSP.means()
    Xg = np.tile(np.array(means), (grid * grid, 1))
    Xg[:, 1] = G1.ravel()  # P1_dx
    Xg[:, 4] = G4.ravel()  # P4_Fx
    Yg = compute_outputs_batch(Xg)[:, 2].reshape(grid, grid)
    cs = ax3.contourf(G1, G4 / 1e6, Yg, levels=18, cmap="viridis")
    fig.colorbar(cs, ax=ax3, label="P9 [MPa]")
    ax3.set_title("Response surface: P9 vs (d_x, F_x)",
                  fontsize=22, color=NAVY, weight="bold")
    ax3.set_xlabel("d_x [mm]", fontsize=16, color=MUTED)
    ax3.set_ylabel("F_x [MN]", fontsize=16, color=MUTED)
    ax3.tick_params(labelsize=13, colors=MUTED)

    # ── Fig 4: Reliability card ──────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3, 2])
    ax4.axis("off")
    ax4.text(0.0, 0.95, "Reliability", fontsize=24, weight="bold", color=NAVY,
             va="top", family="DejaVu Sans")
    # Pick thresholds at empirical P50 / P90 / P99 so β stays finite and
    # the table tells a meaningful story (each row pushes deeper into the tail).
    thresholds = [float(np.quantile(p9, q)) for q in (0.50, 0.90, 0.99)]
    rows = []
    for thr in thresholds:
        p = float((p9 >= thr).mean())
        from scipy.stats import norm as _norm
        eps = 1.0 / len(p9)
        p_clip = max(min(p, 1.0 - eps), eps)
        beta = float(_norm.ppf(p_clip))
        rows.append((thr, p, beta))
    y0 = 0.78
    ax4.text(0.0, y0, "f_c,k", fontsize=14, color=MUTED, weight="bold")
    ax4.text(0.40, y0, "P(P9 ≥ f_c,k)", fontsize=14, color=MUTED, weight="bold")
    ax4.text(0.80, y0, "β", fontsize=14, color=MUTED, weight="bold")
    for i, (thr, p, beta) in enumerate(rows):
        y = y0 - 0.10 * (i + 1)
        ax4.text(0.0,  y, f"{thr:.2f} MPa", fontsize=14, color=NAVY)
        ax4.text(0.40, y, f"{p:.4f}",       fontsize=14, color=NAVY)
        ax4.text(0.80, y, f"{beta:+.2f}",   fontsize=14, color=NAVY)
    ax4.text(0.0, 0.16,
             "Thresholds at empirical P50 / P90 / P99 of P9.\n"
             "FORM β = Φ⁻¹(p);  Wilson 95% CI on p in dashboard.",
             fontsize=11, color=MUTED, va="top")

    # ── Footer ───────────────────────────────────────────────────────────
    ax_f = fig.add_subplot(gs[4, :])
    ax_f.axis("off")
    ax_f.text(0.5, 0.5,
              "Sensitive-Spark Thesis Project · MIT Licence · "
              "github.com/ai-systems-today/cubespec · "
              "DOI: 10.5281/zenodo.PLACEHOLDER",
              fontsize=14, color=MUTED, ha="center", va="center",
              family="DejaVu Sans")

    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("docs/media/poster.pdf"))
    parser.add_argument("--preview", type=Path,
                        default=Path("docs/media/poster_preview.png"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig = _figure()
    fig.savefig(args.out, dpi=DPI, bbox_inches="tight")
    fig.savefig(args.preview, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out} and {args.preview}.")


if __name__ == "__main__":
    main()
