"""
Merge all group small-lexicon annotation CSVs into one file.

Output:
  annotation/all_annotations_combined.csv

Each row is one row from an annotator sheet; `annotation_source` identifies the file.
Rows are stacked in a stable order (Darragh, Josu, Numidia, molina).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ANN = PROJECT / "annotation"

SOURCES: list[tuple[str, Path]] = [
    ("Darragh_annotations.csv", ANN / "Darragh_annotations.csv"),
    ("Josu_annotation.csv", ANN / "Josu_annotation.csv"),
    ("Numidia_annotations-def.csv", ANN / "Numidia_annotations-def.csv"),
    (
        "molina_preannotated_small_conf2plus.csv",
        ANN / "molina_preannotated_small_conf2plus - preannotated_small_conf2plus.csv",
    ),
]

OUT = ANN / "all_annotations_combined.csv"


def main() -> None:
    parts: list[pd.DataFrame] = []
    for label, path in SOURCES:
        if not path.exists():
            raise FileNotFoundError(f"Missing annotation file: {path}")
        df = pd.read_csv(path, dtype={"sentence_id": "Int64"}, low_memory=False)
        df.insert(0, "annotation_source", label)
        parts.append(df)

    combined = pd.concat(parts, ignore_index=True, sort=False)
    combined.to_csv(OUT, index=False)
    print(f"Wrote {len(combined)} rows from {len(parts)} files -> {OUT}")


if __name__ == "__main__":
    main()
