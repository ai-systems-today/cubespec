#!/usr/bin/env python3
"""Emit a JSON manifest of every Markdown doc source with metadata.

Output: dist/docs/manifest.json

Schema per entry:
    path, bytes, sha256, sha256_short, word_count, language, batch
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"

# Map directory prefixes → batch labels (best-effort, for the QA audit)
BATCH_RULES = [
    ("docs/manual/gr/", "B5"),
    ("docs/manual/", "B4"),
    ("docs/tutorials/gr/", "B5"),
    ("docs/tutorials/", "B4"),
    ("docs/instructor/", "B5"),
    ("docs/dev/", "B6"),
    ("docs/defence/", "B6"),
    ("docs/reference/", "B3"),
    ("docs/paper/", "B7"),
    ("docs/media/", "B7"),
]


def detect_lang(rel: str, text: str) -> str:
    if "/gr/" in rel or rel.endswith("-gr.md") or "-gr." in rel:
        return "gr"
    # Heuristic: presence of Greek characters
    if re.search(r"[\u0370-\u03FF\u1F00-\u1FFF]", text):
        # Could be bilingual (e.g., contributions.md, thesis-abstract.md)
        if re.search(r"\b(the|and|of|with)\b", text, re.IGNORECASE):
            return "bilingual"
        return "gr"
    return "en"


def detect_batch(rel: str) -> str:
    for prefix, label in BATCH_RULES:
        if rel.startswith(prefix):
            return label
    return "B7"


def word_count(text: str) -> int:
    # Strip code fences and HTML comments before counting
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    return len(re.findall(r"\S+", text))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "dist" / "docs" / "manifest.json"))
    args = ap.parse_args()

    entries: List[Dict] = []
    for md in sorted(DOCS.rglob("*.md")):
        rel = md.relative_to(REPO).as_posix()
        text = md.read_text(encoding="utf-8", errors="replace")
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        entries.append(
            {
                "path": rel,
                "bytes": md.stat().st_size,
                "sha256": sha,
                "sha256_short": sha[:12],
                "word_count": word_count(text),
                "language": detect_lang(rel, text),
                "batch": detect_batch(rel),
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"count": len(entries), "files": entries}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[manifest] wrote {out} with {len(entries)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
