#!/usr/bin/env python3
"""Build documentation bundles for cubespec.

Concatenates Markdown sources into per-language bundles and renders them to
DOCX (via pandoc) and PDF (via LibreOffice converting the DOCX).

Idempotent. Gracefully degrades when optional renderers are missing.

Usage:
    python python/scripts/build_docs.py --out dist/docs

Outputs (per bundle):
    dist/docs/<bundle>.md
    dist/docs/<bundle>.docx   (if pandoc available)
    dist/docs/<bundle>.pdf    (if pandoc + libreoffice available)
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format="[build_docs] %(levelname)s %(message)s")
log = logging.getLogger("build_docs")

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"


@dataclass
class Bundle:
    name: str
    title: str
    language: str  # "en" | "gr"
    sources: List[Path] = field(default_factory=list)


def _md(*relpaths: str) -> List[Path]:
    out: List[Path] = []
    for rel in relpaths:
        p = DOCS / rel
        if p.exists():
            out.append(p)
        else:
            log.warning("missing source: %s", rel)
    return out


def build_bundles() -> List[Bundle]:
    manual_en = sorted((DOCS / "manual").glob("[0-9][0-9]-*.md"))
    manual_gr = sorted((DOCS / "manual" / "gr").glob("[0-9][0-9]-*.md"))
    tut_en = sorted((DOCS / "tutorials").glob("[0-9][0-9]-*.md"))
    tut_gr = sorted((DOCS / "tutorials" / "gr").glob("[0-9][0-9]-*.md"))
    dev = sorted((DOCS / "dev").glob("*.md"))

    defence_en = _md(
        "defence/slides-outline-en.md",
        "defence/qa-prep-en.md",
        "defence/demo-script-en.md",
        "defence/contributions.md",
    )
    defence_gr = _md(
        "defence/slides-outline-gr.md",
        "defence/qa-prep-gr.md",
        "defence/demo-script-gr.md",
        "defence/contributions.md",
    )

    thesis_en = (
        _md("thesis-abstract.md", "thesis-paper-full-en.md")
        + manual_en
        + tut_en
        + dev
        + defence_en
    )
    thesis_gr = (
        _md("thesis-abstract.md", "thesis-paper-full-gr.md")
        + manual_gr
        + tut_gr
        + defence_gr
    )

    return [
        Bundle("cubespec_thesis_en", "CubeSpec — Thesis Bundle (EN)", "en", thesis_en),
        Bundle("cubespec_thesis_gr", "CubeSpec — Thesis Bundle (GR)", "gr", thesis_gr),
        Bundle("cubespec_manual_en", "CubeSpec — User Manual (EN)", "en", manual_en),
        Bundle("cubespec_manual_gr", "CubeSpec — Εγχειρίδιο Χρήστη (GR)", "gr", manual_gr),
        Bundle("cubespec_defence_en", "CubeSpec — Thesis Defence Pack (EN)", "en", defence_en),
        Bundle("cubespec_defence_gr", "CubeSpec — Πακέτο Υποστήριξης (GR)", "gr", defence_gr),
    ]


def concat_markdown(bundle: Bundle, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_md = out_dir / f"{bundle.name}.md"
    parts: List[str] = [f"% {bundle.title}\n", f"% CubeSpec\n", f"% Auto-generated\n\n"]
    parts.append(f"# {bundle.title}\n\n")
    parts.append(
        "_This document is an auto-generated bundle. Sources live in `docs/`._\n\n"
    )
    for src in bundle.sources:
        rel = src.relative_to(REPO)
        parts.append(f"\n\n<!-- source: {rel} -->\n\n")
        parts.append(f"\n\n---\n\n")
        text = src.read_text(encoding="utf-8")
        parts.append(text)
        parts.append("\n")
    out_md.write_text("".join(parts), encoding="utf-8")
    log.info("wrote %s (%d sources, %d bytes)", out_md.name, len(bundle.sources), out_md.stat().st_size)
    return out_md


def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def md_to_docx(md_path: Path, docx_path: Path) -> bool:
    if not have("pandoc"):
        log.warning("pandoc missing — skip DOCX for %s", md_path.name)
        return False
    cmd = [
        "pandoc",
        str(md_path),
        "-f",
        "markdown",
        "-t",
        "docx",
        "--toc",
        "--toc-depth=2",
        "-o",
        str(docx_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        log.info("wrote %s", docx_path.name)
        return True
    except subprocess.CalledProcessError as e:
        log.error("pandoc failed for %s: %s", md_path.name, e.stderr[-400:])
        return False


def docx_to_pdf(docx_path: Path, out_dir: Path) -> bool:
    if not have("libreoffice") and not have("soffice"):
        log.warning("libreoffice missing — skip PDF for %s", docx_path.name)
        return False
    helper = Path("/tmp/run_libreoffice.py")
    base = ["python", str(helper)] if helper.exists() else ["libreoffice"]
    cmd = base + [
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        str(out_dir),
        str(docx_path),
    ]
    try:
        r = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        pdf = out_dir / (docx_path.stem + ".pdf")
        if pdf.exists():
            log.info("wrote %s (%d bytes)", pdf.name, pdf.stat().st_size)
            return True
        log.error("libreoffice produced no PDF: %s", r.stdout[-400:])
        return False
    except subprocess.CalledProcessError as e:
        log.error("libreoffice failed: %s", (e.stderr or e.stdout or "")[-400:])
        return False
    except subprocess.TimeoutExpired:
        log.error("libreoffice timeout for %s", docx_path.name)
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "dist" / "docs"))
    ap.add_argument("--copy-to", default="/mnt/documents", help="Also copy PDFs here")
    ap.add_argument("--skip-pdf", action="store_true")
    ap.add_argument("--skip-docx", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("renderers: pandoc=%s libreoffice=%s", have("pandoc"), have("libreoffice"))

    bundles = build_bundles()
    summary = []
    for b in bundles:
        md = concat_markdown(b, out_dir)
        docx_ok = pdf_ok = False
        if not args.skip_docx:
            docx_ok = md_to_docx(md, out_dir / f"{b.name}.docx")
        if docx_ok and not args.skip_pdf:
            pdf_ok = docx_to_pdf(out_dir / f"{b.name}.docx", out_dir)
        if pdf_ok and args.copy_to:
            dst = Path(args.copy_to)
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out_dir / f"{b.name}.pdf", dst / f"{b.name}.pdf")
            log.info("copied to %s/%s.pdf", dst, b.name)
        summary.append((b.name, len(b.sources), docx_ok, pdf_ok))

    log.info("--- summary ---")
    for name, n, d, p in summary:
        log.info("%-28s sources=%2d  docx=%s  pdf=%s", name, n, d, p)

    return 0


if __name__ == "__main__":
    sys.exit(main())
