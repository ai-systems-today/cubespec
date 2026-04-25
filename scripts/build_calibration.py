"""Phase A — Build the calibrated surrogate.

Generates a physically-grounded synthetic reference dataset of 500 cube tests,
fits a Poly2 ridge regression (degree-2 with interactions, ridge alpha by 5-fold CV),
and persists all coefficients + scaler stats to a JSON artifact that BOTH the
Python package and the TypeScript dashboard can load — guaranteeing parity.

The "ground truth" used here is an enriched concrete-mechanics model:
  E      = 4733 * sqrt(fc')              (ACI-318 §19.2.2)  [MPa]
  fc'    = k * (rho/2400)^2.5            (density coupling, calibrated to 30 MPa @ 2400 kg/m³)
  sigma  = Fx / (dx*dy)
  P9     = sigma - 0.45 * sqrt(Fy^2 + Fz^2)/(dx*dy) * size_factor
  P8     = sigma / E_eff,    E_eff = E * (1 - 0.0002*(rho-2400))
  P7     = P8 * dz
plus ~3% measurement noise on each output.

This is honest synthetic data — the validation report says so explicitly.
Real lab data drops in by replacing reference_cubes.csv.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "src" / "cubespec" / "models"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

PARAM_KEYS = ["P0_rho", "P1_dx", "P2_dy", "P3_dz", "P4_Fx", "P5_Fy", "P6_Fz"]
OUTPUT_KEYS = ["P7_def", "P8_strain", "P9_compressive_strength"]

# Distribution centres + spreads used to generate the reference set.
# Wider than dashboard CSP so the surrogate is well-conditioned across the
# whole operational envelope.
REF_DIST = {
    "P0_rho": (2400.0, 80.0),
    "P1_dx":  (150.0,  1.5),
    "P2_dy":  (150.0,  1.5),
    "P3_dz":  (150.0,  1.5),
    "P4_Fx":  (8.5e5,  1.5e5),
    "P5_Fy":  (50.0,   40.0),
    "P6_Fz":  (50.0,   40.0),
}

NOISE_REL = 0.03  # 3% multiplicative measurement noise


def ground_truth(X: np.ndarray) -> np.ndarray:
    """Physically-grounded enriched model used to GENERATE reference data.

    The dashboard surrogate must learn this from data — it does not get to
    see the formula.
    """
    rho, dx, dy, dz, Fx, Fy, Fz = (X[:, i] for i in range(7))
    A = dx * dy                                     # mm²
    sigma = Fx / A                                  # MPa
    # Density coupling: fc' grows ~rho^2.5 around the nominal.
    rho_factor = (rho / 2400.0) ** 2.5
    # Lateral disturbance (slightly nonlinear in cross-section size).
    size_factor = 1.0 + 0.001 * (np.sqrt(A) - 150.0)
    disturb = 0.45 * np.sqrt(Fy ** 2 + Fz ** 2) / A * size_factor
    P9 = sigma * rho_factor / rho_factor.mean() - disturb  # MPa, "measured"
    # ACI-318 stiffness with density correction.
    fc_pos = np.maximum(P9, 1.0)
    E = 4733.0 * np.sqrt(fc_pos) * (1.0 + 0.0002 * (rho - 2400.0))  # MPa
    P8 = sigma / E
    P7 = P8 * dz
    return np.column_stack([P7, P8, P9])


def generate_reference(n: int = 500, seed: int = 20250101) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    means = np.array([REF_DIST[k][0] for k in PARAM_KEYS])
    sds = np.array([REF_DIST[k][1] for k in PARAM_KEYS])
    X = means + sds * rng.standard_normal(size=(n, 7))
    # Clip degenerate samples (negative geometry, ~0 load).
    X[:, 1:4] = np.clip(X[:, 1:4], 140.0, 160.0)
    X[:, 0] = np.clip(X[:, 0], 2200.0, 2600.0)
    X[:, 4] = np.clip(X[:, 4], 3.0e5, 1.4e6)
    Y_clean = ground_truth(X)
    noise = rng.normal(0.0, NOISE_REL, size=Y_clean.shape)
    Y = Y_clean * (1.0 + noise)
    df = pd.DataFrame(np.column_stack([X, Y_clean, Y]),
                      columns=PARAM_KEYS
                              + [f"{k}_truth" for k in OUTPUT_KEYS]
                              + [f"{k}_meas" for k in OUTPUT_KEYS])
    return df


def fit_per_output(X: np.ndarray, y: np.ndarray, seed: int = 1337):
    """Fit a Poly2-ridge pipeline with 5-fold CV alpha selection.

    Returns (pipeline, metrics_dict, loocv_residuals).
    """
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)),
        ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25), cv=5)),
    ])
    pipe.fit(X, y)
    y_pred_cv = cross_val_predict(pipe, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=seed))
    resid = y - y_pred_cv
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    # Leave-one-out via fast formula not available for ridge after polyfeatures;
    # report 5-fold CV residuals as the honest equivalent.
    return pipe, {"rmse": rmse, "mae": mae, "r2": r2, "alpha": float(pipe.named_steps["ridge"].alpha_)}, resid


def serialise_pipeline(pipe: Pipeline, output_key: str) -> dict:
    """Pull StandardScaler stats + Poly feature powers + Ridge coefs into JSON."""
    scaler: StandardScaler = pipe.named_steps["scale"]
    poly: PolynomialFeatures = pipe.named_steps["poly"]
    ridge = pipe.named_steps["ridge"]
    powers = poly.powers_.tolist()  # shape (n_terms, 7)
    return {
        "output": output_key,
        "input_keys": PARAM_KEYS,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "powers": powers,
        "coef": ridge.coef_.tolist(),
        "intercept": float(ridge.intercept_),
    }


def main():
    print("[1/4] Generating reference dataset (500 rows)…")
    df = generate_reference(n=500)
    csv_path = DATA_DIR / "reference_cubes.csv"
    df.to_csv(csv_path, index=False)
    print(f"      → {csv_path}")

    X = df[PARAM_KEYS].to_numpy()
    metrics = {}
    artifacts = {}
    residuals = {}
    print("[2/4] Fitting Poly2-ridge per output…")
    for k in OUTPUT_KEYS:
        y = df[f"{k}_meas"].to_numpy()
        pipe, m, resid = fit_per_output(X, y)
        metrics[k] = m
        artifacts[k] = serialise_pipeline(pipe, k)
        residuals[k] = resid.tolist()
        print(f"      {k}: RMSE={m['rmse']:.4g}  MAE={m['mae']:.4g}  R²={m['r2']:.4f}  α={m['alpha']:.3g}")

    print("[3/4] Persisting JSON artifact…")
    artifact = {
        "schema_version": 1,
        "kind": "poly2_ridge",
        "trained_on": str(csv_path.name),
        "n_train": len(df),
        "input_keys": PARAM_KEYS,
        "output_keys": OUTPUT_KEYS,
        "metrics_5fold_cv": metrics,
        "models": artifacts,
    }
    out = MODEL_DIR / "poly2_ridge.json"
    out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"      → {out}")

    print("[4/4] Persisting CV residuals (for QQ plots)…")
    res_path = MODEL_DIR / "poly2_ridge_residuals.json"
    res_path.write_text(json.dumps(residuals), encoding="utf-8")
    print(f"      → {res_path}")
    print("Done.")


if __name__ == "__main__":
    main()
