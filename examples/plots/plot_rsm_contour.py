"""RSM contour of P9 over (P1_dx, P4_Fx) with other factors at their means."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from cubespec import DEFAULT_CSP, compute_outputs

OUT = Path(__file__).resolve().parent / "output" / "rsm_contour.png"
OUT.parent.mkdir(exist_ok=True)


def main() -> None:
    csp = DEFAULT_CSP
    base = {k: csp.params[k].mean for k in csp.params}
    dx_mean, dx_sd = csp.params["P1_dx"].mean, csp.params["P1_dx"].sd
    fx_mean, fx_sd = csp.params["P4_Fx"].mean, csp.params["P4_Fx"].sd

    n = 60
    dx_grid = np.linspace(dx_mean - 3 * dx_sd, dx_mean + 3 * dx_sd, n)
    fx_grid = np.linspace(fx_mean - 3 * fx_sd, fx_mean + 3 * fx_sd, n)
    Z = np.empty((n, n))
    for i, fx in enumerate(fx_grid):
        for j, dx in enumerate(dx_grid):
            s = {**base, "P1_dx": dx, "P4_Fx": fx}
            Z[i, j] = compute_outputs(s)["P9_compressive_strength"]

    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=130)
    cs = ax.contourf(dx_grid, fx_grid / 1e6, Z, levels=20, cmap="viridis")
    cl = ax.contour(dx_grid, fx_grid / 1e6, Z, levels=10, colors="white",
                    linewidths=0.6, alpha=0.7)
    ax.clabel(cl, inline=True, fontsize=7, fmt="%.1f")
    cb = fig.colorbar(cs, ax=ax)
    cb.set_label("P9 (MPa)")
    ax.set_xlabel("P1 — dx (mm)")
    ax.set_ylabel("P4 — Fx (MN)")
    ax.set_title("Response surface of P9 over (dx, Fx)")
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
