"""CSV / JSON writers matching the TypeScript dashboard's export schema."""
from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    df.to_csv(path, index=False)


def write_json(obj, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)


def write_summary_md(summary: dict, path: str | Path) -> None:
    lines = ["# CubeSpec run summary", ""]
    for k, v in summary.items():
        lines.append(f"- **{k}**: {v}")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
