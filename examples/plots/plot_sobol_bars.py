"""Grouped bar chart of Sobol S1 vs ST indices on P9."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from cubespec import DEFAULT_CSP, sobol_indices

OUT = Path(__file__).resolve().parent / "output" / "sobol_bars.png"
OUT.parent.mkdir(exist_ok=True)


def main() -> None:
    df = sobol_indices(DEFAULT_CSP, n_base=2048, seed=1337)
    P9 = df[df["output"] == "P9_compressive_strength"].copy()
    P9 = P9.sort_values("ST", ascending=False).reset_index(drop=True)

    factors = P9["factor"].tolist()
    s1 = P9["S1"].clip(lower=0).to_numpy()
    st = P9["ST"].clip(lower=0).to_numpy()
    x = np.arange(len(factors))
    w = 0.4

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=130)
    ax.bar(x - w / 2, s1, w, label="S1 (first-order)", color="#3B82F6")
    ax.bar(x + w / 2, st, w, label="ST (total)", color="#0F172A")
    ax.set_xticks(x, factors, rotation=20, ha="right")
    ax.set_ylabel("Sensitivity index")
    ax.set_title("Sobol indices on P9 (Saltelli, N_base = 2048)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(1.0, st.max() * 1.1))
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
