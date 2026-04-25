"""Plot the Monte-Carlo distribution of P9 with mean ± 95% CI overlay."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from cubespec import (
    DEFAULT_CSP, sample_independent, compute_outputs_batch, bootstrap_mean_ci,
)

OUT = Path(__file__).resolve().parent / "output" / "p9_histogram.png"
OUT.parent.mkdir(exist_ok=True)


def main() -> None:
    X = sample_independent(DEFAULT_CSP, n=50_000, seed=1337)
    P9 = compute_outputs_batch(X)[:, 2]
    ci = bootstrap_mean_ci(P9, B=1000, seed=1337)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=130)
    ax.hist(P9, bins=80, color="#3B82F6", edgecolor="white", alpha=0.85)
    ax.axvline(ci.mean, color="#0F172A", lw=2, label=f"mean = {ci.mean:.3f} MPa")
    ax.axvspan(ci.lo, ci.hi, color="#0F172A", alpha=0.12,
               label=f"95% CI [{ci.lo:.3f}, {ci.hi:.3f}]")
    ax.set_xlabel("P9 — compressive strength (MPa)")
    ax.set_ylabel("count")
    ax.set_title(f"Monte-Carlo distribution of P9 (N = {len(P9):,}, seed = 1337)")
    ax.legend(loc="upper right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
