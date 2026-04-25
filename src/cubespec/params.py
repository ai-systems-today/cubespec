"""Cube Specification Parameters (CSP) — mirrors `src/components/dashboard/types.ts`.

The seven inputs (P0..P6) are treated as independent (or correlated) Gaussians
unless overridden by a user-supplied distribution.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List
import csv
from pathlib import Path


PARAM_KEYS: List[str] = [
    "P0_rho",
    "P1_dx",
    "P2_dy",
    "P3_dz",
    "P4_Fx",
    "P5_Fy",
    "P6_Fz",
]

OUTPUT_KEYS: List[str] = [
    "P7_def",
    "P8_strain",
    "P9_compressive_strength",
]


@dataclass
class ParamSpec:
    mean: float
    sd: float
    label: str
    units: str
    group: str  # "Material" | "Geometry" | "Forces"


@dataclass
class CSP:
    """Container for the seven input parameter specifications."""
    params: Dict[str, ParamSpec] = field(default_factory=dict)

    def keys(self) -> List[str]:
        return list(self.params.keys())

    def means(self) -> List[float]:
        return [self.params[k].mean for k in PARAM_KEYS]

    def sds(self) -> List[float]:
        return [self.params[k].sd for k in PARAM_KEYS]

    def to_dict(self) -> Dict[str, Dict[str, float | str]]:
        return {k: asdict(v) for k, v in self.params.items()}


def _default_csp() -> CSP:
    return CSP(params={
        "P0_rho": ParamSpec(2400.0,      35.0,    "P0: ρ (density)", "kg/m³", "Material"),
        "P1_dx":  ParamSpec(149.9797,    0.5209,  "P1: dx",          "mm",    "Geometry"),
        "P2_dy":  ParamSpec(150.8522,    0.8199,  "P2: dy",          "mm",    "Geometry"),
        "P3_dz":  ParamSpec(150.0633,    0.4826,  "P3: dz",          "mm",    "Geometry"),
        "P4_Fx":  ParamSpec(1_000_000.0, 25_000.0,"P4: Fx (load)",   "N",     "Forces"),
        "P5_Fy":  ParamSpec(45.6836,     23.1312, "P5: Fy",          "N",     "Forces"),
        "P6_Fz":  ParamSpec(65.3273,     30.0306, "P6: Fz",          "N",     "Forces"),
    })


DEFAULT_CSP: CSP = _default_csp()


def load_csp_csv(path: str | Path) -> CSP:
    """Load a CSV with columns: key,mean,sd[,label,units,group]."""
    csp = CSP(params={})
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row["key"].strip()
            csp.params[key] = ParamSpec(
                mean=float(row["mean"]),
                sd=float(row["sd"]),
                label=row.get("label", key),
                units=row.get("units", ""),
                group=row.get("group", "Material"),
            )
    # Fill any missing with defaults so the loader is forgiving.
    for k, v in DEFAULT_CSP.params.items():
        csp.params.setdefault(k, v)
    return csp


def save_csp_csv(csp: CSP, path: str | Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "mean", "sd", "label", "units", "group"])
        for k in PARAM_KEYS:
            p = csp.params[k]
            writer.writerow([k, p.mean, p.sd, p.label, p.units, p.group])
