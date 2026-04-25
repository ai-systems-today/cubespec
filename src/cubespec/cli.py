"""Command-line interface: ``cubespec run | doe | sobol``.

Uses argparse so it works without optional dependencies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .params import DEFAULT_CSP, load_csp_csv, OUTPUT_KEYS
from .sampling import sample_independent
from .model import compute_outputs_batch, set_mode
from .doe import full_factorial, fractional_factorial, main_effects
from .sobol import sobol_indices
from .bootstrap import bootstrap_mean_ci


def _load(csp_path: str | None):
    return load_csp_csv(csp_path) if csp_path else DEFAULT_CSP


def cmd_run(args) -> int:
    set_mode(args.model)
    csp = _load(args.csp)
    X = sample_independent(csp, n=args.n, seed=args.seed)
    Y = compute_outputs_batch(X)
    means = {k: float(Y[:, i].mean()) for i, k in enumerate(OUTPUT_KEYS)}
    sds = {k: float(Y[:, i].std(ddof=1)) for i, k in enumerate(OUTPUT_KEYS)}
    ci = bootstrap_mean_ci(Y[:, 2], B=args.bootstrap, seed=args.seed)
    summary = {
        "n": args.n,
        "seed": args.seed,
        "model": args.model,
        "means": means,
        "sds": sds,
        "P9_ci_95": {"lo": ci.lo, "mean": ci.mean, "hi": ci.hi, "B": ci.B},
    }
    out = Path(args.output)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def cmd_doe(args) -> int:
    csp = _load(args.csp)
    if args.design.startswith("fractional"):
        frac = args.design.split("-", 1)[1]
        df = fractional_factorial(csp, fraction=frac)
    else:
        df = full_factorial(csp, levels=args.levels)
    df.to_csv(args.output, index=False)
    eff = main_effects(df)
    print(eff.to_string(index=False))
    return 0


def cmd_sobol(args) -> int:
    csp = _load(args.csp)
    df = sobol_indices(csp, n_base=args.n, seed=args.seed)
    df.to_csv(args.output, index=False)
    print(df.to_string(index=False))
    return 0


def cmd_reliability(args) -> int:
    from .uncertainty import decompose_variance, reliability_index
    set_mode(args.model)
    csp = _load(args.csp)
    X = sample_independent(csp, n=args.n, seed=args.seed)
    Y = compute_outputs_batch(X)
    rel = reliability_index(
        Y, threshold=args.threshold, output=args.output_key, direction=args.direction
    )
    var = decompose_variance(Y)
    summary = {
        "n": args.n, "seed": args.seed, "model": args.model,
        "reliability": rel.as_dict(),
        "variance_decomposition": {
            k: {"aleatory": v.aleatory, "epistemic": v.epistemic, "total": v.total,
                "aleatory_frac": v.aleatory_frac, "epistemic_frac": v.epistemic_frac}
            for k, v in var.items()
        },
    }
    Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def cmd_optimise(args) -> int:
    from .optimise import optimise, default_bounds
    set_mode(args.model)
    csp = _load(args.csp)
    bounds = default_bounds(csp, k_sigma=args.k_sigma)
    result = optimise(
        csp,
        output=args.output_key,
        direction="maximise" if args.maximise else "minimise",
        bounds=bounds,
        seed=args.seed,
        n_starts=args.starts,
    )
    Path(args.output).write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
    print(json.dumps(result.as_dict(), indent=2))
    return 0


def cmd_report(args) -> int:
    from .report import build_report
    out = build_report(
        csp_path=args.csp,
        out_pdf=args.pdf,
        seed=args.seed,
        n=args.n,
        threshold_mpa=args.threshold,
        model=args.model,
    )
    print(f"Wrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cubespec", description="CubeSpec CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="Monte Carlo run + bootstrap CI on P9")
    pr.add_argument("--csp", default=None, help="CSV with CSP overrides")
    pr.add_argument("--n", type=int, default=50_000)
    pr.add_argument("--seed", type=int, default=1337)
    pr.add_argument("--bootstrap", type=int, default=1000)
    pr.add_argument("--model", choices=["calibrated", "analytic"], default="calibrated",
                    help="Surrogate model class. Default: calibrated Poly2-ridge.")
    pr.add_argument("--output", default="report.json")
    pr.set_defaults(func=cmd_run)

    pd_ = sub.add_parser("doe", help="Full or fractional factorial DOE")
    pd_.add_argument("--csp", default=None)
    pd_.add_argument("--design", default="full",
                     choices=["full", "fractional-1/2", "fractional-1/4", "fractional-1/8"])
    pd_.add_argument("--levels", type=int, default=2)
    pd_.add_argument("--output", default="design.csv")
    pd_.set_defaults(func=cmd_doe)

    ps = sub.add_parser("sobol", help="Sobol S1/ST sensitivity (Saltelli scheme)")
    ps.add_argument("--csp", default=None)
    ps.add_argument("--n", type=int, default=1024)
    ps.add_argument("--seed", type=int, default=1337)
    ps.add_argument("--output", default="sobol.csv")
    ps.set_defaults(func=cmd_sobol)

    pl = sub.add_parser("reliability", help="Variance decomposition + reliability β-index for one output")
    pl.add_argument("--csp", default=None)
    pl.add_argument("--n", type=int, default=20_000)
    pl.add_argument("--seed", type=int, default=1337)
    pl.add_argument("--threshold", type=float, required=True,
                    help="Acceptance threshold (units of the chosen output, e.g. 30.0 MPa for P9).")
    pl.add_argument("--output-key", default="P9_compressive_strength",
                    choices=OUTPUT_KEYS)
    pl.add_argument("--direction", choices=["ge", "le"], default="ge",
                    help="'ge' for strength-style (Y ≥ threshold); 'le' for deformation-style.")
    pl.add_argument("--model", choices=["calibrated", "analytic"], default="calibrated")
    pl.add_argument("--output", default="reliability.json")
    pl.set_defaults(func=cmd_reliability)

    po = sub.add_parser("optimise", help="Box-constrained optimum on the calibrated surrogate (L-BFGS-B multi-start)")
    po.add_argument("--csp", default=None)
    po.add_argument("--output-key", default="P9_compressive_strength", choices=OUTPUT_KEYS)
    po.add_argument("--maximise", action="store_true", help="Maximise the target (default minimises).")
    po.add_argument("--k-sigma", type=float, default=3.0, help="Box-bound width in σ around each μ.")
    po.add_argument("--starts", type=int, default=8)
    po.add_argument("--seed", type=int, default=1337)
    po.add_argument("--model", choices=["calibrated", "analytic"], default="calibrated")
    po.add_argument("--output", default="optimise.json")
    po.set_defaults(func=cmd_optimise)

    pr2 = sub.add_parser("report", help="Build the branded 8-page thesis PDF")
    pr2.add_argument("--csp", default=None)
    pr2.add_argument("--pdf", default="cubespec_report.pdf",
                     help="Output PDF path.")
    pr2.add_argument("--seed", type=int, default=1337)
    pr2.add_argument("--n", type=int, default=10_000)
    pr2.add_argument("--threshold", type=float, default=30.0,
                     help="P9 reliability threshold in MPa.")
    pr2.add_argument("--model", choices=["calibrated", "analytic"],
                     default="calibrated")
    pr2.set_defaults(func=cmd_report)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
