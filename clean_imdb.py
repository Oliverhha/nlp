#!/usr/bin/env python3
# clean IMDb reviews

import argparse
import html
import json
import re
import unicodedata
from pathlib import Path

GROUPS = ("train/unsup", "train/pos", "train/neg", "test/pos", "test/neg")


def clean_review(text: str) -> str:
    # preserve punctuation, stop words, and negation, normalize web formatting
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?\s*>", " ", text, flags=re.IGNORECASE)
    text = unicodedata.normalize("NFC", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=base / "aclImdb")
    parser.add_argument("--output-dir", type=Path, default=base / "aclImdb_cleaned")
    args = parser.parse_args()
    source = args.data_dir.resolve()
    destination = args.output_dir.resolve()
    if not source.is_dir():
        parser.error(f"Dataset directory not found: {source}")
    if destination == source or source in destination.parents or destination in source.parents:
        parser.error("Input and output directories must be separate, not nested.")
    if destination.exists():
        parser.error(f"Output already exists; choose a fresh --output-dir: {destination}")
    for group in GROUPS:
        if not (source / group).is_dir():
            parser.error(f"Missing dataset directory: {source / group}")
    destination.mkdir(parents=True)
    statistics = {}
    for group in GROUPS:
        files = sorted((source / group).glob("*.txt"))
        target = destination / group
        target.mkdir(parents=True)
        counts = {"input": len(files), "written": 0, "empty_removed": 0,
                  "changed": 0, "html_break_before": 0, "html_break_after": 0}
        for path in files:
            raw = path.read_text(encoding="utf-8")
            cleaned = clean_review(raw)
            counts["html_break_before"] += int(bool(re.search(r"<br\s*/?\s*>", raw, re.I)))
            counts["html_break_after"] += int(bool(re.search(r"<br\s*/?\s*>", cleaned, re.I)))
            counts["changed"] += int(raw != cleaned)
            if not cleaned:
                counts["empty_removed"] += 1
                continue
            (target / path.name).write_text(cleaned, encoding="utf-8")
            counts["written"] += 1
        statistics[group] = counts
        print(f"{group}: {counts}")
    report = {"source": str(source), "output": str(destination),
              "normalization": "HTML unescape; HTML breaks to spaces; NFC; lowercase; collapse whitespace",
              "groups": statistics}
    (destination / "cleaning_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved cleaned reviews and summary to: {destination}")


if __name__ == "__main__":
    main()
