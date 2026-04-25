"""Phase D — branded 8-page thesis report (Python parity).

Mirrors ``src/components/dashboard/pdf/ThesisReport.tsx`` page-for-page so
the CLI and the frontend produce structurally identical artifacts. Uses
ReportLab (no system font dependency, lightweight install).

Public API:

    build_report(csp_path, out_pdf, seed, n) -> Path
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from .model import compute_outputs_batch, set_mode
from .params import DEFAULT_CSP, OUTPUT_KEYS, PARAM_KEYS, load_csp_csv
from .sampling import sample_independent
from .uncertainty import decompose_variance, reliability_index

# Brand palette (matches React-pdf styles).
INK = colors.HexColor("#0F172A")
BODY = colors.HexColor("#1E293B")
MUTED = colors.HexColor("#475569")
RULE = colors.HexColor("#CBD5E1")
ACCENT = colors.HexColor("#0D9488")
ACCENT_SOFT = colors.HexColor("#CCFBF1")
CARD = colors.HexColor("#F8FAFC")
WHITE = colors.HexColor("#FFFFFF")

# Report constants
TOTAL_PAGES = 8
MARGIN = 18 * mm


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    out: dict[str, ParagraphStyle] = {}
    out["h1"] = ParagraphStyle("h1", parent=base["Heading1"],
                               textColor=INK, fontName="Helvetica-Bold",
                               fontSize=18, spaceAfter=2)
    out["h2"] = ParagraphStyle("h2", parent=base["Heading2"],
                               textColor=INK, fontName="Helvetica-Bold",
                               fontSize=13, spaceBefore=12, spaceAfter=4)
    out["h3"] = ParagraphStyle("h3", parent=base["Heading3"],
                               textColor=BODY, fontName="Helvetica-Bold",
                               fontSize=11, spaceBefore=8, spaceAfter=2)
    out["body"] = ParagraphStyle("body", parent=base["BodyText"],
                                 textColor=BODY, fontName="Helvetica",
                                 fontSize=10, leading=14, spaceAfter=4)
    out["caption"] = ParagraphStyle("caption", parent=base["BodyText"],
                                    textColor=MUTED, fontName="Helvetica",
                                    fontSize=9, leading=12)
    out["bullet"] = ParagraphStyle("bullet", parent=out["body"],
                                   leftIndent=14, bulletIndent=2,
                                   bulletFontName="Helvetica-Bold",
                                   bulletColor=ACCENT)
    out["coverTitle"] = ParagraphStyle("coverTitle", parent=base["Title"],
                                       textColor=WHITE,
                                       fontName="Helvetica-Bold",
                                       fontSize=32, leading=36, spaceAfter=12)
    out["coverSub"] = ParagraphStyle("coverSub", parent=base["BodyText"],
                                     textColor=colors.HexColor("#CBD5E1"),
                                     fontName="Helvetica",
                                     fontSize=14, leading=18, spaceAfter=24)
    out["coverMeta"] = ParagraphStyle("coverMeta", parent=base["BodyText"],
                                      textColor=colors.HexColor("#94A3B8"),
                                      fontName="Helvetica",
                                      fontSize=10, leading=14)
    return out


def _accent_rule(width_mm: float = 22) -> Table:
    t = Table([[""]], colWidths=[width_mm * mm], rowHeights=[3])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), ACCENT)]))
    return t


def _table(data: list[list[str]], col_widths: Optional[list[float]] = None,
           head_bg: colors.Color = CARD) -> Table:
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), head_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), INK),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, RULE),
        ("LINEABOVE", (0, 0), (-1, 0), 1, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("FONTNAME", (1, 1), (-1, -1), "Courier"),
    ]))
    return t


def _draw_footer(cnv: canvas.Canvas, page_num: int) -> None:
    cnv.saveState()
    cnv.setFont("Helvetica", 9)
    cnv.setFillColor(MUTED)
    cnv.drawString(MARGIN, 14 * mm, "CubeSpec · Thesis Report")
    cnv.drawRightString(A4[0] - MARGIN, 14 * mm, f"{page_num} / {TOTAL_PAGES}")
    cnv.restoreState()


def _draw_cover_bg(cnv: canvas.Canvas) -> None:
    cnv.saveState()
    cnv.setFillColor(INK)
    cnv.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
    cnv.setFillColor(ACCENT)
    cnv.rect(MARGIN, A4[1] - 30 * mm, 22 * mm, 4, fill=1, stroke=0)
    cnv.restoreState()


def _fmt(v: float, d: int = 3) -> str:
    if not np.isfinite(v):
        return "—"
    return f"{v:.{d}f}"


def build_report(
    csp_path: Optional[str | Path],
    out_pdf: str | Path,
    seed: int = 1337,
    n: int = 50_000,
    threshold_mpa: float = 30.0,
    model: str = "calibrated",
) -> Path:
    """Generate the 8-page thesis report PDF.

    Parameters
    ----------
    csp_path : path or None
        CSV with CSP overrides; defaults to the bundled DEFAULT_CSP.
    out_pdf : path
        Destination PDF path.
    seed, n : int
        Monte Carlo seed and sample count.
    threshold_mpa : float
        Reliability threshold for P9 (default = Eurocode-2 C25/30 = 30 MPa).
    model : "calibrated" | "analytic"
        Surrogate mode used for sampling.
    """
    set_mode(model)  # type: ignore[arg-type]
    csp = load_csp_csv(csp_path) if csp_path else DEFAULT_CSP

    # Run Monte Carlo + analyses.
    X = sample_independent(csp, n=n, seed=seed)
    Y = compute_outputs_batch(X)
    var = decompose_variance(Y)
    rel = reliability_index(Y, threshold=threshold_mpa,
                            output="P9_compressive_strength", direction="ge")

    # Per-output summary stats.
    summary_rows = []
    for i, k in enumerate(OUTPUT_KEYS):
        col = Y[:, i]
        summary_rows.append({
            "output": k,
            "mean": float(col.mean()),
            "sd": float(col.std(ddof=1)),
            "p5": float(np.quantile(col, 0.05)),
            "p95": float(np.quantile(col, 0.95)),
        })

    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    # Lazily load calibration metrics from the artifact JSON.
    try:
        from .surrogate import load_artifact
        art = load_artifact()
        metrics = art.get("metrics_5fold_cv", {})
        meta = {"kind": art.get("kind", model),
                "trainedOn": art.get("trained_on", "n/a"),
                "nTrain": int(art.get("n_train", 0))}
    except Exception:
        metrics = {k: {"r2": float("nan"), "rmse": float("nan"),
                       "mae": float("nan"), "alpha": float("nan")}
                   for k in OUTPUT_KEYS}
        meta = {"kind": model, "trainedOn": "n/a", "nTrain": 0}

    iso_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    S = _styles()

    # ---- Document setup ---------------------------------------------------
    doc = BaseDocTemplate(
        str(out_pdf), pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN + 8, bottomMargin=MARGIN + 8,
        title="CubeSpec — Thesis Report",
        author="CubeSpec",
        subject="Monte Carlo · DOE · Sensitivity · Reliability",
    )
    cover_frame = Frame(MARGIN, MARGIN, A4[0] - 2 * MARGIN,
                        A4[1] - 2 * MARGIN, id="cover", showBoundary=0)
    body_frame = Frame(MARGIN, MARGIN + 10, A4[0] - 2 * MARGIN,
                       A4[1] - 2 * MARGIN - 10, id="body", showBoundary=0)

    # Counters via doc attribute so PageTemplate callbacks can use them.
    doc._page_index = 0  # type: ignore[attr-defined]

    def cover_on_page(cnv, _doc):
        _draw_cover_bg(cnv)
        # No footer on cover.

    def body_on_page(cnv, doc_):
        doc_._page_index += 1  # type: ignore[attr-defined]
        _draw_footer(cnv, doc_._page_index + 1)  # cover = page 1

    doc.addPageTemplates([
        PageTemplate(id="cover", frames=[cover_frame], onPage=cover_on_page),
        PageTemplate(id="body", frames=[body_frame], onPage=body_on_page),
    ])

    story = []

    # ---- Page 1: Cover ----------------------------------------------------
    story.append(Spacer(1, 60 * mm))
    story.append(Paragraph("CubeSpec", S["coverTitle"]))
    story.append(Paragraph(
        "Statistical materials testing — Monte Carlo, DOE, sensitivity, "
        "and reliability for the 150 mm cube compressive test.",
        S["coverSub"]))
    cover_meta_table = Table([
        [Paragraph("<font color='#94A3B8'>Generated</font>", S["coverMeta"]),
         Paragraph(f"<font color='#FFFFFF' size='14'>{iso_date}</font>", S["coverMeta"])],
        [Paragraph("<font color='#94A3B8'>Seed</font>", S["coverMeta"]),
         Paragraph(f"<font color='#FFFFFF' size='14'>{seed}</font>", S["coverMeta"])],
        [Paragraph("<font color='#94A3B8'>Surrogate</font>", S["coverMeta"]),
         Paragraph(
             f"<font color='#FFFFFF' size='12'>"
             f"Poly2-ridge (R²={_fmt(metrics['P9_compressive_strength']['r2'])}, "
             f"n={meta['nTrain']})</font>", S["coverMeta"])],
    ], colWidths=[40 * mm, None])
    cover_meta_table.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(cover_meta_table)
    story.append(Spacer(1, 24 * mm))
    story.append(Paragraph(
        f"<font color='#94A3B8'>"
        f"docs/validation-report.md · seed={seed} · N={n:,}</font>",
        S["coverMeta"]))

    # Switch to body template for the remaining pages.
    story.append(PageBreak())
    from reportlab.platypus import NextPageTemplate
    story.insert(len(story) - 0, NextPageTemplate("body"))

    # ---- Page 2: ToC ------------------------------------------------------
    story.append(Paragraph("Table of contents", S["h1"]))
    story.append(_accent_rule())
    story.append(Spacer(1, 8))
    toc_data = [["#", "Section", "Page"]] + [
        ["1.", "Cover", "1"],
        ["2.", "Table of contents", "2"],
        ["3.", "Methods", "3"],
        ["4.", "Calibration & variance", "4"],
        ["5.", "Output distribution (P9)", "5"],
        ["6.", "Global sensitivity (Sobol)", "6"],
        ["7.", "Reliability", "7"],
        ["8.", "References", "8"],
    ]
    story.append(_table(toc_data, col_widths=[14 * mm, 110 * mm, 22 * mm]))
    story.append(PageBreak())

    # ---- Page 3: Methods --------------------------------------------------
    story.append(Paragraph("Methods", S["h1"]))
    story.append(_accent_rule())
    story.append(Paragraph("Inputs (CSP)", S["h2"]))
    story.append(Paragraph(
        "Seven uncertain inputs P0–P6 model material density, cube geometry, "
        "and loading. Each is sampled as N(μ, σ); a Cholesky decomposition is "
        "used when a correlation matrix is supplied.", S["body"]))
    csp_data = [["Param", "Label", "μ", "σ", "Units"]]
    for k in PARAM_KEYS:
        spec = csp.params[k]
        csp_data.append([k, spec.label, _fmt(spec.mean, 2),
                         _fmt(spec.sd, 2), spec.units])
    story.append(_table(csp_data,
                        col_widths=[20 * mm, 60 * mm, 25 * mm, 25 * mm, 20 * mm]))
    story.append(Paragraph("Surrogate", S["h2"]))
    story.append(Paragraph(
        f"Calibrated Poly2-ridge regression trained on {meta['nTrain']} "
        "reference cubes (5-fold CV; see docs/validation-report.md). The "
        "analytic σ = F/A baseline is retained as an interpretability fallback.",
        S["body"]))
    story.append(Paragraph("Pipeline", S["h2"]))
    for line in [
        "Sample CSP → calibrated surrogate evaluation",
        "Convergence: μ ± 1.96·SE plus a percentile bootstrap (B = 1000)",
        "Variance-based Sobol S1 + ST via the Saltelli scheme",
        "Variance decomposition: aleatory (sampling) + epistemic (CV residual)",
        "Reliability: empirical exceedance + Wilson 95% CI + FORM β-index",
    ]:
        story.append(Paragraph(line, S["bullet"], bulletText="•"))
    story.append(PageBreak())

    # ---- Page 4: Calibration & Variance -----------------------------------
    story.append(Paragraph("Calibration & variance", S["h1"]))
    story.append(_accent_rule())
    story.append(Paragraph("5-fold CV metrics", S["h2"]))
    cv_data = [["Output", "R²", "RMSE", "MAE", "α (ridge)"]]
    for k in OUTPUT_KEYS:
        m = metrics.get(k, {})
        cv_data.append([k, _fmt(m.get("r2", float("nan"))),
                        _fmt(m.get("rmse", float("nan")), 4),
                        _fmt(m.get("mae", float("nan")), 4),
                        _fmt(m.get("alpha", float("nan")), 4)])
    story.append(_table(cv_data,
                        col_widths=[55 * mm, 25 * mm, 25 * mm, 25 * mm, 25 * mm]))

    story.append(Paragraph("Variance decomposition", S["h2"]))
    var_data = [["Output", "Aleatory", "Epistemic", "Total", "Aleatory %"]]
    for k, v in var.items():
        var_data.append([
            k,
            _fmt(v.aleatory, 4), _fmt(v.epistemic, 4),
            _fmt(v.total, 4), f"{100 * v.aleatory_frac:.1f}%",
        ])
    story.append(_table(var_data,
                        col_widths=[55 * mm, 25 * mm, 30 * mm, 25 * mm, 20 * mm]))
    story.append(PageBreak())

    # ---- Page 5: P9 distribution -----------------------------------------
    story.append(Paragraph("Output distribution · P9", S["h1"]))
    story.append(_accent_rule())
    story.append(Paragraph(
        f"Empirical distribution of compressive strength (MPa) over "
        f"N = {n:,} Monte Carlo runs. Summary statistics across the three "
        "outputs are shown below.", S["body"]))
    sum_data = [["Output", "Mean", "SD", "P5", "P95"]]
    for r in summary_rows:
        sum_data.append([r["output"], _fmt(r["mean"], 4), _fmt(r["sd"], 4),
                         _fmt(r["p5"], 4), _fmt(r["p95"], 4)])
    story.append(_table(sum_data,
                        col_widths=[55 * mm, 30 * mm, 30 * mm, 25 * mm, 25 * mm]))
    story.append(PageBreak())

    # ---- Page 6: Sobol ----------------------------------------------------
    story.append(Paragraph("Global sensitivity", S["h1"]))
    story.append(_accent_rule())
    story.append(Paragraph(
        "Variance-based Sobol indices (Saltelli scheme): S1 measures first-"
        "order contribution; ST captures total effect including interactions. "
        "Valid under independent inputs.", S["body"]))
    try:
        from .sobol import sobol_indices
        sob = sobol_indices(csp, n_base=256, seed=seed)
        sob_data = [["Factor", "Output", "S1", "ST"]]
        for _, row in sob.iterrows():
            sob_data.append([row["factor"], row["output"],
                             _fmt(float(row["S1"]), 4),
                             _fmt(float(row["ST"]), 4)])
        story.append(_table(sob_data,
                            col_widths=[30 * mm, 65 * mm, 25 * mm, 25 * mm]))
    except Exception as exc:  # pragma: no cover
        story.append(Paragraph(f"<i>(Sobol computation unavailable: {exc})</i>",
                               S["caption"]))
    story.append(PageBreak())

    # ---- Page 7: Reliability ----------------------------------------------
    story.append(Paragraph("Reliability", S["h1"]))
    story.append(_accent_rule())
    story.append(Paragraph(
        f"Reliability against compressive-strength threshold τ = "
        f"{threshold_mpa:g} MPa (Eurocode-2 C25/30 characteristic strength). "
        "Empirical exceedance with Wilson 95% CI and the FORM β-index Φ⁻¹(p).",
        S["body"]))

    rel_card = Table([
        [
            Paragraph(
                f"<b>P(P9 ≥ τ)</b><br/>"
                f"<font size='22' color='#0D9488'>"
                f"{rel.p * 100:.2f}%</font><br/>"
                f"<font color='#475569' size='9'>Wilson 95% CI: "
                f"[{rel.p_lo * 100:.2f}%, {rel.p_hi * 100:.2f}%]</font><br/>"
                f"<font color='#475569' size='9'>"
                f"{rel.successes:,} / {rel.n:,} runs satisfy.</font>",
                S["body"]),
            Paragraph(
                f"<b>FORM β-index</b><br/>"
                f"<font size='22' color='#0F172A'>"
                f"{('∞' if not np.isfinite(rel.beta) else f'{rel.beta:.3f}')}"
                f"</font><br/>"
                f"<font color='#475569' size='9'>β = Φ⁻¹(p)</font>",
                S["body"]),
        ]
    ], colWidths=[80 * mm, 80 * mm])
    rel_card.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CARD),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(Spacer(1, 8))
    story.append(rel_card)
    story.append(PageBreak())

    # ---- Page 8: References -----------------------------------------------
    story.append(Paragraph("References", S["h1"]))
    story.append(_accent_rule())
    refs = [
        "EN 12390-3:2019 — Testing hardened concrete, Part 3: Compressive strength of test specimens.",
        "Eurocode 2 (EN 1992-1-1) — Design of concrete structures.",
        "Saltelli, A. et al. (2008) Global Sensitivity Analysis: The Primer. Wiley.",
        "Sobol', I. M. (2001) Global sensitivity indices for nonlinear mathematical models. Math. Comp. Sim.",
        "Herman, J. & Usher, W. (2017) SALib: Sensitivity analysis library in Python. JOSS.",
        "Hoerl & Kennard (1970) Ridge Regression. Technometrics.",
        "Hasofer & Lind (1974) Exact and invariant second-moment code format. ASCE.",
        "Harris, C. R. et al. (2020) Array programming with NumPy. Nature.",
        "Virtanen, P. et al. (2020) SciPy 1.0. Nature Methods.",
    ]
    for i, r in enumerate(refs, 1):
        story.append(Paragraph(r, S["bullet"], bulletText=f"{i}."))

    doc.build(story)
    return out_pdf


__all__ = ["build_report"]
