"""
Inter-annotator agreement (Cohen's κ), human vs LLM agreement, and descriptive stats
for the small-conf2plus group annotation CSVs.

Assumptions:
  - Everyone completed the *first 10 data rows* in each annotator file (same items if
    sentence_id order matches; we use intersection of sentence_id in those first 10 rows).
  - LLM reference labels come from annotation/preannotated_small_conf2plus.csv
    (merged on sentence_id + target_word).

Outputs (metrics/):
  - group_kappa_pairwise_first10.csv
  - group_human_vs_llm_by_annotator.csv
  - group_human_vs_llm_pooled.csv
  - group_descriptive_analytics.csv
  - group_essay_topics_by_annotator.csv
  - group_essay_topics_summary.csv
  - group_essay_topics_pooled.csv
  - group_cefr_band_counts.csv
"""

from __future__ import annotations

import itertools
from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_DIR = PROJECT_ROOT / "annotation"
METRICS_DIR = PROJECT_ROOT / "metrics"

ANNOTATOR_FILES: dict[str, Path] = {
    "Darragh": ANNOTATION_DIR / "Darragh_annotations.csv",
    "Josu": ANNOTATION_DIR / "Josu_annotation.csv",
    "Numidia": ANNOTATION_DIR / "Numidia_annotations-def.csv",
    "molina": ANNOTATION_DIR
    / "molina_preannotated_small_conf2plus - preannotated_small_conf2plus.csv",
}

LLM_REFERENCE = ANNOTATION_DIR / "preannotated_small_conf2plus.csv"

HUMAN_REG = "register_markedness (human)"
HUMAN_SUB = "substitution_naturalness (human)"
HUMAN_CTX = "context_label (human)"
HUMAN_CONF = "confidence (human)"
LLM_REG = "register_markedness_llm"
LLM_SUB = "substitution_natural_llm"
LLM_CONF = "confidence_llm"


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"sentence_id": "Int64"}, low_memory=False)


def to_numeric_binary(series: pd.Series) -> pd.Series:
    """0/1 float with NaN for missing."""
    s = pd.to_numeric(series, errors="coerce")
    return s


def to_numeric_optional(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def cohen_kappa_binary(y1: np.ndarray, y2: np.ndarray) -> tuple[float, int]:
    """
    Cohen's kappa for binary (0/1) labels. NaNs must be pre-filtered or passed as mask.
    Returns (kappa, n_valid).
    """
    mask = ~(np.isnan(y1) | np.isnan(y2))
    n = int(mask.sum())
    if n < 2:
        return float("nan"), n
    a = y1[mask].astype(int)
    b = y2[mask].astype(int)
    # restrict to {0,1}
    if not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all():
        # treat any non-0/1 as nan pair
        ok = (np.isin(a, [0, 1])) & (np.isin(b, [0, 1]))
        n = int(ok.sum())
        if n < 2:
            return float("nan"), n
        a, b = a[ok], b[ok]
    # κ is undefined / not informative if either rater is constant
    if np.unique(a).size < 2 or np.unique(b).size < 2:
        return float("nan"), int(len(a))
    po = float(np.mean(a == b))
    p1 = float(np.mean(a))
    p2 = float(np.mean(b))
    pe = p1 * p2 + (1.0 - p1) * (1.0 - p2)
    if abs(1.0 - pe) < 1e-12:
        return float("nan"), n
    return (po - pe) / (1.0 - pe), n


def cohen_kappa_nominal(y1: pd.Series, y2: pd.Series) -> tuple[float, int]:
    """Cohen's kappa for string nominal labels (case-insensitive, stripped)."""
    mask = y1.notna() & y2.notna()
    s1 = y1[mask].astype(str).str.strip().str.lower()
    s2 = y2[mask].astype(str).str.strip().str.lower()
    bad = {"nan", "", "none"}
    ok = ~s1.isin(bad) & ~s2.isin(bad)
    s1, s2 = s1[ok], s2[ok]
    n = len(s1)
    if n < 2:
        return float("nan"), n
    po = float((s1 == s2).mean())
    cats = pd.unique(pd.concat([s1, s2], ignore_index=True))
    pe = 0.0
    for c in cats:
        pe += float((s1 == c).mean() * (s2 == c).mean())
    if abs(1.0 - pe) < 1e-12:
        return float("nan"), n
    return (po - pe) / (1.0 - pe), n


def agreement_rate(y1: np.ndarray, y2: np.ndarray) -> tuple[float, int]:
    mask = ~(np.isnan(y1) | np.isnan(y2))
    n = int(mask.sum())
    if n == 0:
        return float("nan"), 0
    return float(np.mean(y1[mask] == y2[mask])), n


def row_key_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "target_word" not in out.columns:
        raise ValueError("expected column target_word")
    out["_key"] = (
        out["sentence_id"].astype(str) + "|" + out["target_word"].astype(str)
    )
    return out


def first_n_overlap_keys(dfs: dict[str, pd.DataFrame], n: int = 10) -> set[str]:
    key_sets: list[set[str]] = []
    for df in dfs.values():
        sub = df.head(n)
        key_sets.append(set(row_key_cols(sub)["_key"]))
    return set.intersection(*key_sets) if key_sets else set()


def load_llm_reference() -> pd.DataFrame:
    ref = _read_csv(LLM_REFERENCE)
    ref = row_key_cols(ref)
    keep = [
        "_key",
        "sentence_id",
        "target_word",
        LLM_REG,
        LLM_SUB,
        LLM_CONF,
        "l1",
        "cefr",
        "cefr_numeric",
        "nationality",
        "grade",
        "wordcount",
    ]
    cols = [c for c in keep if c in ref.columns]
    return ref[cols]


def human_vs_llm_metrics(
    human_reg: pd.Series,
    human_sub: pd.Series,
    llm_reg: pd.Series,
    llm_sub: pd.Series,
) -> dict:
    hr = to_numeric_binary(human_reg).to_numpy(dtype=float)
    hs = to_numeric_binary(human_sub).to_numpy(dtype=float)
    lr = to_numeric_binary(llm_reg).to_numpy(dtype=float)
    ls = to_numeric_binary(llm_sub).to_numpy(dtype=float)
    k_reg, n_reg = cohen_kappa_binary(hr, lr)
    k_sub, n_sub = cohen_kappa_binary(hs, ls)
    acc_reg, _ = agreement_rate(hr, lr)
    acc_sub, _ = agreement_rate(hs, ls)
    m_r = ~(np.isnan(hr) | np.isnan(lr))
    m_s = ~(np.isnan(hs) | np.isnan(ls))
    return {
        "n_pairs_register": n_reg,
        "n_pairs_substitution": n_sub,
        "kappa_register_vs_llm": k_reg,
        "kappa_substitution_vs_llm": k_sub,
        "accuracy_register_vs_llm": acc_reg,
        "accuracy_substitution_vs_llm": acc_sub,
        "human_register_variance_on_merged": bool(np.unique(hr[m_r]).size >= 2) if m_r.any() else False,
        "llm_register_variance_on_merged": bool(np.unique(lr[m_r]).size >= 2) if m_r.any() else False,
        "human_substitution_variance_on_merged": bool(np.unique(hs[m_s]).size >= 2) if m_s.any() else False,
        "llm_substitution_variance_on_merged": bool(np.unique(ls[m_s]).size >= 2) if m_s.any() else False,
    }


def descriptive_block(df: pd.DataFrame, annotator: str) -> dict:
    """Per-annotator descriptives on rows with any human register filled."""
    d = df.copy()
    reg = to_numeric_binary(d[HUMAN_REG])
    sub = to_numeric_binary(d[HUMAN_SUB])
    conf = to_numeric_optional(d[HUMAN_CONF]) if HUMAN_CONF in d.columns else pd.Series(np.nan)
    cefr_n = to_numeric_optional(d["cefr_numeric"]) if "cefr_numeric" in d.columns else pd.Series(np.nan)

    mask = reg.notna()
    sub_mask = sub.notna()
    conf_mask = conf.notna()

    out = {
        "annotator": annotator,
        "n_rows_file": len(d),
        "n_rows_human_register_filled": int(mask.sum()),
        "n_rows_human_substitution_filled": int(sub_mask.sum()),
        "mean_register_markedness_human": float(reg[mask].mean()) if mask.any() else float("nan"),
        "mean_substitution_naturalness_human": float(sub[sub_mask].mean())
        if sub_mask.any()
        else float("nan"),
        "mean_confidence_human": float(conf[conf_mask].mean()) if conf_mask.any() else float("nan"),
        "mean_cefr_numeric": float(cefr_n[mask].mean()) if mask.any() and cefr_n.notna().any() else float("nan"),
    }
    # Mode / top categories for L1, nationality, CEFR band
    for col in ("l1", "nationality", "cefr"):
        if col in d.columns and mask.any():
            vc = d.loc[mask, col].astype(str).str.strip()
            vc = vc[vc.str.len() > 0]
            if len(vc):
                top = vc.value_counts().head(3)
                out[f"top_{col}_1"] = top.index[0] if len(top) > 0 else ""
                out[f"top_{col}_1_count"] = int(top.iloc[0]) if len(top) > 0 else 0
            else:
                out[f"top_{col}_1"] = ""
                out[f"top_{col}_1_count"] = 0
    return out


def export_essay_topic_and_cefr_analytics(
    annotators: dict[str, pd.DataFrame], metrics_dir: Path
) -> None:
    """
    Essay `topic` is a categorical task label (e.g. 'Summarizing a story').
    Summaries use the *mode* (most common topic) per annotator — there is no numeric
    'average topic'; we also report distribution and mean CEFR numeric per topic slice.
    """
    detail_rows: list[dict] = []
    summary_rows: list[dict] = []
    cefr_rows: list[dict] = []
    pool_topics: list[str] = []

    for ann, df in annotators.items():
        d = df.copy()
        reg = to_numeric_binary(d[HUMAN_REG])
        mask = reg.notna()
        sub = d.loc[mask]
        if len(sub) == 0:
            summary_rows.append(
                {
                    "annotator": ann,
                    "n_human_labeled_rows": 0,
                    "most_common_essay_topic": "",
                    "n_rows_in_most_common_topic": 0,
                    "pct_in_most_common_topic": float("nan"),
                    "n_distinct_topics": 0,
                    "mean_cefr_numeric_overall": float("nan"),
                }
            )
            continue

        if "topic" not in sub.columns:
            topics = pd.Series(["(no topic column)"] * len(sub))
        else:
            topics = sub["topic"].fillna("").astype(str).str.strip()
            topics = topics.replace("", "(missing topic)")

        pool_topics.extend(topics.tolist())

        if "cefr" in sub.columns:
            for cefr, cnt in sub["cefr"].astype(str).str.strip().value_counts().items():
                cefr_rows.append({"annotator": ann, "cefr_band": cefr, "n_rows": int(cnt)})

        vc = topics.value_counts()
        total = int(len(sub))
        cefr_all = (
            float(to_numeric_optional(sub["cefr_numeric"]).mean())
            if "cefr_numeric" in sub.columns
            else float("nan")
        )

        for topic, cnt in vc.items():
            tmask = topics == topic
            tdf = sub.loc[tmask]
            detail_rows.append(
                {
                    "annotator": ann,
                    "essay_topic": topic,
                    "n_human_labeled_rows": int(cnt),
                    "pct_of_annotator_labels": round(100.0 * cnt / total, 1),
                    "mean_register_markedness": float(
                        to_numeric_binary(tdf[HUMAN_REG]).mean()
                    ),
                    "mean_substitution_naturalness": float(
                        to_numeric_binary(tdf[HUMAN_SUB]).mean()
                    )
                    if HUMAN_SUB in tdf.columns
                    else float("nan"),
                    "mean_cefr_numeric": float(
                        to_numeric_optional(tdf["cefr_numeric"]).mean()
                    )
                    if "cefr_numeric" in tdf.columns
                    else float("nan"),
                }
            )

        mode_topic = str(vc.index[0])
        summary_rows.append(
            {
                "annotator": ann,
                "n_human_labeled_rows": total,
                "most_common_essay_topic": mode_topic,
                "n_rows_in_most_common_topic": int(vc.iloc[0]),
                "pct_in_most_common_topic": round(100.0 * vc.iloc[0] / total, 1),
                "n_distinct_topics": int(topics.nunique()),
                "mean_cefr_numeric_overall": cefr_all,
            }
        )

    detail_df = pd.DataFrame(detail_rows)
    if not detail_df.empty:
        detail_df = detail_df.sort_values(
            ["annotator", "n_human_labeled_rows"], ascending=[True, False]
        )
    detail_df.to_csv(metrics_dir / "group_essay_topics_by_annotator.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(
        metrics_dir / "group_essay_topics_summary.csv", index=False
    )

    if pool_topics:
        pool_series = pd.Series(pool_topics)
        pvc = pool_series.value_counts()
        tot = int(pvc.sum())
        pool_df = pd.DataFrame(
            {
                "essay_topic": pvc.index.astype(str),
                "n_rows_pooled_human_labels": pvc.values.astype(int),
                "pct_of_pooled": np.round(100.0 * pvc.values / tot, 1),
            }
        )
        pool_df.to_csv(metrics_dir / "group_essay_topics_pooled.csv", index=False)

    pd.DataFrame(cefr_rows).to_csv(
        metrics_dir / "group_cefr_band_counts.csv", index=False
    )


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    annotators: dict[str, pd.DataFrame] = {}
    for name, path in ANNOTATOR_FILES.items():
        if not path.exists():
            raise FileNotFoundError(path)
        annotators[name] = _read_csv(path)

    llm = load_llm_reference()
    llm_indexed = llm.set_index("_key")

    # --- First-10 overlap: pairwise kappa (canonical = first 10 rows of LLM reference) ---
    ref_canon = _read_csv(LLM_REFERENCE)
    overlap_keys = set(row_key_cols(ref_canon.head(10))["_key"])
    alt = first_n_overlap_keys(annotators, n=10)
    if len(overlap_keys) < 10 and alt:
        # If reference has fewer than 10 rows, fall back to annotator intersection
        overlap_keys = alt

    pairwise_rows: list[dict] = []
    names = list(annotators.keys())
    for col_human, label in [
        (HUMAN_REG, "register_markedness"),
        (HUMAN_SUB, "substitution_naturalness"),
    ]:
        # build aligned arrays per key
        per_key: dict[str, dict[str, float]] = {k: {} for k in overlap_keys}
        for ann in names:
            dfk = row_key_cols(annotators[ann])
            sub = dfk[dfk["_key"].isin(overlap_keys)]
            for _, row in sub.iterrows():
                k = row["_key"]
                v = to_numeric_binary(pd.Series([row[col_human]])).iloc[0]
                if pd.notna(v):
                    per_key[k][ann] = float(v)

        for a, b in itertools.combinations(names, 2):
            y1, y2 = [], []
            for k in sorted(overlap_keys):
                if a in per_key[k] and b in per_key[k]:
                    y1.append(per_key[k][a])
                    y2.append(per_key[k][b])
            if len(y1) >= 2:
                kappa, n = cohen_kappa_binary(np.array(y1), np.array(y2))
                acc, _ = agreement_rate(np.array(y1), np.array(y2))
            else:
                kappa, n, acc = float("nan"), len(y1), float("nan")
            pairwise_rows.append(
                {
                    "subset": "first_10_reference_rows",
                    "variable": label,
                    "rater_a": a,
                    "rater_b": b,
                    "n_overlapping_items": n,
                    "cohens_kappa": kappa,
                    "percent_agreement": acc * 100 if acc == acc else float("nan"),
                }
            )

    # context_label nominal kappa on the same overlap set (once, not per binary variable)
    per_key_ctx: dict[str, dict[str, str]] = {k: {} for k in overlap_keys}
    for ann in names:
        dfk = row_key_cols(annotators[ann])
        sub = dfk[dfk["_key"].isin(overlap_keys)]
        for _, row in sub.iterrows():
            k = row["_key"]
            raw = row.get(HUMAN_CTX, np.nan)
            if pd.notna(raw) and str(raw).strip():
                per_key_ctx[k][ann] = str(raw).strip()
    for a, b in itertools.combinations(names, 2):
        s1, s2 = [], []
        for k in sorted(overlap_keys):
            if a in per_key_ctx[k] and b in per_key_ctx[k]:
                s1.append(per_key_ctx[k][a])
                s2.append(per_key_ctx[k][b])
        if len(s1) >= 2:
            kappa, n = cohen_kappa_nominal(pd.Series(s1), pd.Series(s2))
            acc = float(np.mean(np.array(s1) == np.array(s2)))
        else:
            kappa, n, acc = float("nan"), len(s1), float("nan")
        pairwise_rows.append(
            {
                "subset": "first_10_reference_rows",
                "variable": "context_label",
                "rater_a": a,
                "rater_b": b,
                "n_overlapping_items": n,
                "cohens_kappa": kappa,
                "percent_agreement": acc * 100 if acc == acc else float("nan"),
            }
        )

    pd.DataFrame(pairwise_rows).to_csv(
        METRICS_DIR / "group_kappa_pairwise_first10.csv", index=False
    )

    # --- Each annotator vs LLM (all rows with both sides present) ---
    by_ann_rows: list[dict] = []
    pooled_hr, pooled_lr, pooled_hs, pooled_ls = [], [], [], []

    for ann, df in annotators.items():
        dfk = row_key_cols(df)
        merge_cols = ["_key", HUMAN_REG, HUMAN_SUB]
        merge_cols = [c for c in merge_cols if c in dfk.columns]
        left = dfk[merge_cols]
        merged = left.merge(
            llm_indexed[[LLM_REG, LLM_SUB]].reset_index(),
            on="_key",
            how="inner",
        )
        m = human_vs_llm_metrics(
            merged[HUMAN_REG],
            merged[HUMAN_SUB],
            merged[LLM_REG],
            merged[LLM_SUB],
        )
        m["annotator"] = ann
        m["n_rows_merged"] = len(merged)
        by_ann_rows.append(m)

        hr = to_numeric_binary(merged[HUMAN_REG]).to_numpy(dtype=float)
        hs = to_numeric_binary(merged[HUMAN_SUB]).to_numpy(dtype=float)
        lr = to_numeric_binary(merged[LLM_REG]).to_numpy(dtype=float)
        ls = to_numeric_binary(merged[LLM_SUB]).to_numpy(dtype=float)
        mask_r = ~(np.isnan(hr) | np.isnan(lr))
        mask_s = ~(np.isnan(hs) | np.isnan(ls))
        pooled_hr.extend(hr[mask_r].tolist())
        pooled_lr.extend(lr[mask_r].tolist())
        pooled_hs.extend(hs[mask_s].tolist())
        pooled_ls.extend(ls[mask_s].tolist())

    pd.DataFrame(by_ann_rows).to_csv(
        METRICS_DIR / "group_human_vs_llm_by_annotator.csv", index=False
    )

    pooled = {
        "scope": "all_annotators_pooled_vs_llm",
        "n_register_pairs": len(pooled_hr),
        "n_substitution_pairs": len(pooled_hs),
        "kappa_register": cohen_kappa_binary(np.array(pooled_hr), np.array(pooled_lr))[0],
        "kappa_substitution": cohen_kappa_binary(np.array(pooled_hs), np.array(pooled_ls))[0],
        "accuracy_register": agreement_rate(np.array(pooled_hr), np.array(pooled_lr))[0],
        "accuracy_substitution": agreement_rate(np.array(pooled_hs), np.array(pooled_ls))[0],
    }
    pd.DataFrame([pooled]).to_csv(
        METRICS_DIR / "group_human_vs_llm_pooled.csv", index=False
    )

    # --- Descriptive analytics ---
    desc_rows = [descriptive_block(df, ann) for ann, df in annotators.items()]
    # Corpus-level from LLM reference
    ref = _read_csv(LLM_REFERENCE)
    reg_llm = to_numeric_binary(ref[LLM_REG]) if LLM_REG in ref.columns else pd.Series(dtype=float)
    sub_llm = to_numeric_binary(ref[LLM_SUB]) if LLM_SUB in ref.columns else pd.Series(dtype=float)
    desc_rows.append(
        {
            "annotator": "LLM_reference_file",
            "n_rows_file": len(ref),
            "n_rows_human_register_filled": int(reg_llm.notna().sum()),
            "n_rows_human_substitution_filled": int(sub_llm.notna().sum()),
            "mean_register_markedness_human": float("nan"),
            "mean_substitution_naturalness_human": float("nan"),
            "mean_confidence_human": float("nan"),
            "mean_cefr_numeric": float(to_numeric_optional(ref["cefr_numeric"]).mean())
            if "cefr_numeric" in ref.columns
            else float("nan"),
            "mean_register_markedness_llm": float(reg_llm.mean()) if reg_llm.notna().any() else float("nan"),
            "mean_substitution_natural_llm": float(sub_llm.mean()) if sub_llm.notna().any() else float("nan"),
            "mean_confidence_llm": float(to_numeric_optional(ref[LLM_CONF]).mean())
            if LLM_CONF in ref.columns
            else float("nan"),
        }
    )
    pd.DataFrame(desc_rows).to_csv(
        METRICS_DIR / "group_descriptive_analytics.csv", index=False
    )

    export_essay_topic_and_cefr_analytics(annotators, METRICS_DIR)

    # Console summary
    print("Wrote:")
    for f in (
        "group_kappa_pairwise_first10.csv",
        "group_human_vs_llm_by_annotator.csv",
        "group_human_vs_llm_pooled.csv",
        "group_descriptive_analytics.csv",
        "group_essay_topics_by_annotator.csv",
        "group_essay_topics_summary.csv",
        "group_essay_topics_pooled.csv",
        "group_cefr_band_counts.csv",
    ):
        print(f"  {METRICS_DIR / f}")
    print(f"\nOverlap keys (reference first 10 or fallback): {len(overlap_keys)} items")
    print("(Cohen's kappa / percent agreement in group_kappa_pairwise_first10.csv)")


if __name__ == "__main__":
    main()
