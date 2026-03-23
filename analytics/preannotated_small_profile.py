"""
Corpus-level profile of LLM pre-annotations in preannotated_small.csv.

Outputs under metrics/:
  - preannotated_small_overall.csv       — % over-formal, n rows, etc.
  - preannotated_small_register_label_dist.csv
  - preannotated_small_confidence_dist.csv
  - preannotated_small_lemma_stats.csv   — per-lemma flags & confidence
  - preannotated_small_lexicon_token_counts.csv — small lexicon latinate types + row counts (0 if absent)
  - preannotated_small_l1_by_overformal.csv — L1 × LLM over-formal flag counts & %
  - preannotated_small_cefr_by_overformal.csv — CEFR band × over-formal counts & %

Input (first path that exists):
  - approach_a/output/preannotated_small.csv
  - Filtering_LLM_Filtering/output/preannotated_small.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
METRICS = PROJECT / "metrics"


def resolve_preannotated_path() -> Path:
    candidates = [
        PROJECT / "approach_a" / "output" / "preannotated_small.csv",
        PROJECT
        / "Filtering_LLM_Filtering"
        / "output"
        / "preannotated_small.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "preannotated_small.csv not found at:\n"
        + "\n".join(f"  - {c}" for c in candidates)
    )


def load_small_lexicon_tokens(project: Path) -> pd.DataFrame:
    """Latinate + germanic from Small_seed_lexicon.csv in file order."""
    lex_path = (
        project
        / "annotation"
        / "lexicon"
        / "lexicon"
        / "Small_seed_lexicon.csv"
    )
    if not lex_path.exists():
        raise FileNotFoundError(f"Small lexicon not found: {lex_path}")
    lex = pd.read_csv(lex_path)
    if "latinate" not in lex.columns:
        raise ValueError(f"Expected 'latinate' in {lex_path}")
    lex = lex.copy()
    lex["latinate"] = lex["latinate"].astype(str).str.strip().str.lower()
    if "germanic" in lex.columns:
        lex["germanic"] = lex["germanic"].astype(str).str.strip().str.lower()
    else:
        lex["germanic"] = ""
    return lex[["latinate", "germanic"]].reset_index(drop=True)


CEFR_LEVEL_ORDER = ["A1", "A2", "B1", "B2", "C1"]


def crosstab_overformal(
    df: pd.DataFrame,
    group_col: str,
    category_order: list[str] | None = None,
    sort_by_n: bool = False,
) -> pd.DataFrame:
    """Counts of register_markedness_llm 0 vs 1 per group (e.g. l1 or cefr)."""
    if group_col not in df.columns:
        return pd.DataFrame(
            columns=[
                group_col,
                "n_not_overformal_llm_0",
                "n_overformal_llm_1",
                "n_total",
                "pct_overformal_llm",
            ]
        )
    sub = df[[group_col, "register_markedness_llm"]].copy()
    sub[group_col] = sub[group_col].fillna("(missing)").astype(str).str.strip()
    sub = sub.replace({group_col: {"": "(missing)"}})
    rows_out = []
    for key, g in sub.groupby(group_col, sort=False):
        c0 = int((g["register_markedness_llm"] == 0).sum())
        c1 = int((g["register_markedness_llm"] == 1).sum())
        tot = c0 + c1
        rows_out.append(
            {
                group_col: key,
                "n_not_overformal_llm_0": c0,
                "n_overformal_llm_1": c1,
                "n_total": tot,
                "pct_overformal_llm": round(100.0 * c1 / tot, 2) if tot else 0.0,
            }
        )
    out = pd.DataFrame(rows_out)
    if category_order:
        order_map = {lab: i for i, lab in enumerate(category_order)}
        out["_ord"] = out[group_col].map(lambda x: order_map.get(x, 999))
        out = out.sort_values("_ord").drop(columns="_ord")
    elif sort_by_n:
        out = out.sort_values("n_total", ascending=False)
    return out.reset_index(drop=True)


def main() -> None:
    METRICS.mkdir(parents=True, exist_ok=True)
    path = resolve_preannotated_path()
    df = pd.read_csv(path, low_memory=False)

    df["register_markedness_llm"] = pd.to_numeric(
        df.get("register_markedness_llm", 0), errors="coerce"
    ).fillna(0).astype(int)
    df["confidence_llm"] = pd.to_numeric(
        df.get("confidence_llm", 1), errors="coerce"
    ).fillna(1).astype(int)

    n = len(df)
    n_marked = int((df["register_markedness_llm"] == 1).sum())
    pct_marked = round(100.0 * n_marked / n, 2) if n else 0.0

    overall = pd.DataFrame(
        [
            {
                "source_file": str(path.relative_to(PROJECT)),
                "n_rows": n,
                "n_register_marked_llm_1": n_marked,
                "n_register_marked_llm_0": n - n_marked,
                "pct_over_formal_llm": pct_marked,
                "mean_confidence_llm": float(df["confidence_llm"].mean()),
                "mean_confidence_when_marked": float(
                    df.loc[df["register_markedness_llm"] == 1, "confidence_llm"].mean()
                )
                if n_marked
                else float("nan"),
                "mean_confidence_when_unmarked": float(
                    df.loc[df["register_markedness_llm"] == 0, "confidence_llm"].mean()
                )
                if n - n_marked
                else float("nan"),
            }
        ]
    )
    overall.to_csv(METRICS / "preannotated_small_overall.csv", index=False)

    # register_label distribution
    if "register_label" in df.columns:
        rl = df["register_label"].fillna("(missing)").astype(str).str.strip()
        rl = rl.replace("", "(missing)")
        vc = rl.value_counts()
        reg_dist = pd.DataFrame(
            {
                "register_label": vc.index,
                "count": vc.values,
                "pct_of_rows": np.round(100.0 * vc.values / n, 2),
            }
        )
    else:
        reg_dist = pd.DataFrame(columns=["register_label", "count", "pct_of_rows"])
    reg_dist.to_csv(METRICS / "preannotated_small_register_label_dist.csv", index=False)

    # confidence distribution
    vc_c = df["confidence_llm"].value_counts().sort_index()
    conf_dist = pd.DataFrame(
        {
            "confidence_llm": vc_c.index.astype(int),
            "count": vc_c.values.astype(int),
            "pct_of_rows": np.round(100.0 * vc_c.values / n, 2),
        }
    )
    conf_dist.to_csv(METRICS / "preannotated_small_confidence_dist.csv", index=False)

    # L1 and CEFR vs LLM over-formal flag
    l1_tab = crosstab_overformal(df, "l1", sort_by_n=True)
    l1_tab.to_csv(METRICS / "preannotated_small_l1_by_overformal.csv", index=False)
    cefr_tab = crosstab_overformal(df, "cefr", category_order=CEFR_LEVEL_ORDER)
    cefr_tab.to_csv(METRICS / "preannotated_small_cefr_by_overformal.csv", index=False)

    # Per lemma (target_word)
    if "target_word" not in df.columns:
        raise ValueError("expected column target_word")
    tw = df["target_word"].astype(str).str.strip().str.lower()
    df = df.assign(_lemma=tw)

    rows = []
    for lemma, sub in df.groupby("_lemma"):
        tot = len(sub)
        flagged = int((sub["register_markedness_llm"] == 1).sum())
        rows.append(
            {
                "target_word": lemma,
                "n_rows": tot,
                "n_flagged_overformal": flagged,
                "pct_flagged": round(100.0 * flagged / tot, 2) if tot else 0.0,
                "mean_confidence_llm": float(sub["confidence_llm"].mean()),
                "mean_confidence_when_flagged": float(
                    sub.loc[sub["register_markedness_llm"] == 1, "confidence_llm"].mean()
                )
                if flagged
                else float("nan"),
            }
        )
    lemma_stats = pd.DataFrame(rows).sort_values("n_rows", ascending=False)
    lemma_stats.to_csv(METRICS / "preannotated_small_lemma_stats.csv", index=False)

    # Small lexicon only: every latinate type with count in this file (0 = no hits)
    count_by_lemma = lemma_stats.set_index("target_word")["n_rows"].to_dict()
    lex_df = load_small_lexicon_tokens(PROJECT)
    lex_rows = []
    for i, row in lex_df.iterrows():
        lat = row["latinate"]
        if not lat or lat == "nan":
            continue
        cnt = int(count_by_lemma.get(lat, 0))
        lex_rows.append(
            {
                "lexicon_order": i + 1,
                "latinate": lat,
                "germanic_pair": row["germanic"],
                "n_rows_in_preannotated": cnt,
                "pct_of_preannotated_rows": round(100.0 * cnt / n, 4) if n else 0.0,
            }
        )
    pd.DataFrame(lex_rows).to_csv(
        METRICS / "preannotated_small_lexicon_token_counts.csv", index=False
    )

    # Rankings as separate small tables for plotting / papers
    by_flag_count = lemma_stats.sort_values(
        ["n_flagged_overformal", "pct_flagged"], ascending=False
    )
    by_flag_rate = lemma_stats[lemma_stats["n_rows"] >= 5].sort_values(
        "pct_flagged", ascending=False
    )
    by_mean_conf = lemma_stats.sort_values("mean_confidence_llm", ascending=False)

    by_flag_count.head(20).to_csv(
        METRICS / "preannotated_small_lemmas_top_flagged_count.csv", index=False
    )
    by_flag_rate.head(20).to_csv(
        METRICS / "preannotated_small_lemmas_top_flagged_rate.csv", index=False
    )
    by_mean_conf.head(20).to_csv(
        METRICS / "preannotated_small_lemmas_top_mean_confidence.csv", index=False
    )

    print("Wrote:")
    for name in (
        "preannotated_small_overall.csv",
        "preannotated_small_register_label_dist.csv",
        "preannotated_small_confidence_dist.csv",
        "preannotated_small_lemma_stats.csv",
        "preannotated_small_lemmas_top_flagged_count.csv",
        "preannotated_small_lemmas_top_flagged_rate.csv",
        "preannotated_small_lemmas_top_mean_confidence.csv",
        "preannotated_small_lexicon_token_counts.csv",
        "preannotated_small_l1_by_overformal.csv",
        "preannotated_small_cefr_by_overformal.csv",
    ):
        print(f"  {METRICS / name}")
    print(f"\nSource: {path}")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
