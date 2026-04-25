"""Measurement-CSV ingestion — mirrors the dashboard's DOE measurements parser.

The dashboard accepts a CSV with one row per replicate per design point and
the following flexible schema (case-insensitive, columns may appear in any
order):

  - run | row | id                    REQUIRED — the design-point index
  - strength_mpa | strength | p9 |    OPTIONAL — measured P9 (MPa)
    p9_compressive_strength | mpa |
    value
  - p8_strain | p8 | strain           OPTIONAL — measured P8 strain
  - p7_def | p7 | deformation | def   OPTIONAL — measured P7 deformation (mm)

Replicate columns ending in ``_1``, ``_2`` are detected and averaged. If
multiple rows share the same ``run`` they are also averaged.

The output ``MeasurementSet`` is the same shape used by every notebook and
by ``cubespec.report`` so the comparison with predicted values is identical
in the dashboard, the notebooks and any downstream script.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import csv
import re

import numpy as np
import pandas as pd


_RUN_KEYS = {"run", "row", "id"}
_STRENGTH_KEYS = {
    "strength_mpa", "strength", "p9", "p9_compressive_strength", "mpa", "value",
}
_STRAIN_KEYS = {"p8_strain", "p8", "strain"}
_DEF_KEYS = {"p7_def", "p7", "deformation", "def"}

_REPLICATE_SUFFIX = re.compile(r"_(\d+)$")


@dataclass
class MeasurementRecord:
    """Per-design-point average of all replicates supplied for that run."""
    run: int
    strength: Optional[float] = None
    strain: Optional[float] = None
    deformation: Optional[float] = None
    replicates: int = 0


@dataclass
class MeasurementSet:
    """Parsed CSV ready to compare against predicted values."""
    records: Dict[int, MeasurementRecord] = field(default_factory=dict)
    source: Optional[str] = None
    columns_detected: Dict[str, str] = field(default_factory=dict)

    def runs(self) -> list[int]:
        return sorted(self.records.keys())

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for run in self.runs():
            r = self.records[run]
            rows.append({
                "run": r.run,
                "P9_compressive_strength_meas": r.strength,
                "P8_strain_meas": r.strain,
                "P7_def_meas": r.deformation,
                "replicates": r.replicates,
            })
        return pd.DataFrame(rows)

    def align(self, predicted: pd.DataFrame, output: str = "P9_compressive_strength") -> pd.DataFrame:
        """Inner-join measurements with a predicted DataFrame on ``run``.

        ``predicted`` must contain a ``run`` column and the chosen ``output``
        column (one of ``P7_def``, ``P8_strain``, ``P9_compressive_strength``).
        Returns a DataFrame with columns ``run``, ``predicted``, ``measured``,
        ``residual`` (measured − predicted) and ``replicates``.
        """
        if "run" not in predicted.columns:
            raise ValueError("predicted DataFrame must contain a 'run' column")
        if output not in predicted.columns:
            raise ValueError(f"predicted DataFrame missing output column {output!r}")
        meas_col = {
            "P9_compressive_strength": "P9_compressive_strength_meas",
            "P8_strain": "P8_strain_meas",
            "P7_def": "P7_def_meas",
        }[output]
        joined = predicted[["run", output]].merge(
            self.to_dataframe()[["run", meas_col, "replicates"]],
            on="run", how="inner",
        )
        joined = joined.rename(columns={output: "predicted", meas_col: "measured"})
        joined = joined.dropna(subset=["measured"])
        joined["residual"] = joined["measured"] - joined["predicted"]
        return joined.reset_index(drop=True)


def _classify(stem: str) -> Optional[str]:
    if stem in _STRENGTH_KEYS:
        return "strength"
    if stem in _STRAIN_KEYS:
        return "strain"
    if stem in _DEF_KEYS:
        return "deformation"
    return None


def parse_measurements(source: Union[str, Path]) -> MeasurementSet:
    """Parse a measurements CSV. Mirrors the dashboard parser exactly.

    Raises ``ValueError`` with a clear message if the CSV lacks a run column
    or has no recognised measurement columns — the same errors the DOE tab
    surfaces as toast notifications.
    """
    path = Path(source)
    text = path.read_text(encoding="utf-8")
    reader = csv.reader(text.splitlines())
    rows = [r for r in reader if any(c.strip() for c in r)]
    if len(rows) < 2:
        raise ValueError(f"{path}: file is empty or has no data rows")

    header = [c.strip().lower() for c in rows[0]]
    try:
        run_idx = next(i for i, h in enumerate(header) if h in _RUN_KEYS)
    except StopIteration as exc:
        raise ValueError(
            f"{path}: missing required 'run' column (also accepted: row, id)"
        ) from exc

    detected: Dict[str, str] = {}
    col_map: list[Tuple[int, str]] = []
    for c, h in enumerate(header):
        if c == run_idx:
            continue
        stem = _REPLICATE_SUFFIX.sub("", h)
        field_name = _classify(stem)
        if field_name is not None:
            col_map.append((c, field_name))
            detected.setdefault(field_name, h)

    if not col_map:
        raise ValueError(
            f"{path}: no recognised measurement columns. Accepted stems: "
            f"strength_mpa | strength | p9 | strain | p8 | deformation | p7 "
            f"(with optional _1, _2 replicate suffixes)."
        )

    accumulator: Dict[int, Dict[str, list[float]]] = {}
    for raw in rows[1:]:
        cell = raw[run_idx].strip() if run_idx < len(raw) else ""
        try:
            run = int(cell)
        except ValueError:
            continue
        bucket = accumulator.setdefault(run, {"strength": [], "strain": [], "deformation": []})
        for c, field_name in col_map:
            if c >= len(raw):
                continue
            try:
                v = float(raw[c].strip())
            except (ValueError, AttributeError):
                continue
            if not np.isfinite(v):
                continue
            bucket[field_name].append(v)

    records: Dict[int, MeasurementRecord] = {}
    for run, bucket in accumulator.items():
        reps = max(len(bucket["strength"]), len(bucket["strain"]), len(bucket["deformation"]))
        records[run] = MeasurementRecord(
            run=run,
            strength=float(np.mean(bucket["strength"])) if bucket["strength"] else None,
            strain=float(np.mean(bucket["strain"])) if bucket["strain"] else None,
            deformation=float(np.mean(bucket["deformation"])) if bucket["deformation"] else None,
            replicates=reps,
        )

    return MeasurementSet(records=records, source=str(path), columns_detected=detected)


def write_template(path: Union[str, Path], n_runs: int = 8) -> Path:
    """Write a blank measurements template the user can fill in.

    Useful for handing to a lab partner: drop the file in, fill the
    measurement columns, save as CSV.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        fh.write("run,strength_mpa_1,strength_mpa_2,p8_strain,p7_def\n")
        for r in range(1, n_runs + 1):
            fh.write(f"{r},,,,\n")
    return p


__all__ = [
    "MeasurementRecord",
    "MeasurementSet",
    "parse_measurements",
    "write_template",
]
