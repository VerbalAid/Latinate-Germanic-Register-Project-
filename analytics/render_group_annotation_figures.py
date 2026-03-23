"""
Build charts for docs/group_annotation_interpretation.md from metrics/*.csv.

Run after:
  python analytics/group_annotation_agreement.py

Then:
  python analytics/render_group_annotation_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
METRICS = PROJECT / "metrics"
FIG_DIR = PROJECT / "docs" / "figures"


def _pairwise_kappa_matrix(kappa: pd.DataFrame, variable: str) -> tuple[np.ndarray, list[str]]:
    """Symmetric matrix of Cohen's kappa for variable (e.g. register_markedness)."""
    sub = kappa[kappa["variable"] == variable].copy()
    names = sorted(
        set(sub["rater_a"].unique()) | set(sub["rater_b"].unique())
    )
    n = len(names)
    idx = {name: i for i, name in enumerate(names)}
    mat = np.full((n, n), np.nan, dtype=float)
    np.fill_diagonal(mat, 1.0)
    for _, row in sub.iterrows():
        i, j = idx[row["rater_a"]], idx[row["rater_b"]]
        v = row["cohens_kappa"]
        mat[i, j] = v
        mat[j, i] = v
    return mat, names


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
        }
    )

    by_ann = pd.read_csv(METRICS / "group_human_vs_llm_by_annotator.csv")
    kappa = pd.read_csv(METRICS / "group_kappa_pairwise_first10.csv")
    desc = pd.read_csv(METRICS / "group_descriptive_analytics.csv")
    pooled = pd.read_csv(METRICS / "group_human_vs_llm_pooled.csv")
    topics_pool = pd.read_csv(METRICS / "group_essay_topics_pooled.csv")
    topics_sum = pd.read_csv(METRICS / "group_essay_topics_summary.csv")
    cefr_counts = pd.read_csv(METRICS / "group_cefr_band_counts.csv")

    w = 0.35

    # --- Fig 1: Human vs AI match rate ---
    names = by_ann["annotator"].tolist()
    reg_acc = (by_ann["accuracy_register_vs_llm"] * 100).round(0).astype(int)
    sub_acc = (by_ann["accuracy_substitution_vs_llm"] * 100).round(0).astype(int)
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        [i - w / 2 for i in x],
        reg_acc,
        width=w,
        label="“Over-formal?” (same as AI)",
        color="#2E86AB",
    )
    ax.bar(
        [i + w / 2 for i in x],
        sub_acc,
        width=w,
        label="“Natural simpler word?” (same as AI)",
        color="#E94F37",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 100)
    ax.set_ylabel("How often answer matched the AI (%)")
    ax.set_title("Each annotator vs the AI — agreement on two questions")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.35)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_human_vs_ai_agreement.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Fig 2: Annotator tendencies ---
    hum = desc[desc["annotator"] != "LLM_reference_file"].copy()
    ann_names = hum["annotator"].tolist()
    marked_pct = (hum["mean_register_markedness_human"] * 100).round(0).astype(int)
    natural_pct = (hum["mean_substitution_naturalness_human"] * 100).round(0).astype(int)
    x2 = range(len(ann_names))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        [i - w / 2 for i in x2],
        marked_pct,
        width=w,
        label="Said “yes, over-formal” (avg)",
        color="#6A4C93",
    )
    ax.bar(
        [i + w / 2 for i in x2],
        natural_pct,
        width=w,
        label="Said “yes, simpler word sounds natural” (avg)",
        color="#1982C4",
    )
    ax.set_xticks(list(x2))
    ax.set_xticklabels(ann_names)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Average “yes” rate on filled-in rows (%)")
    ax.set_title("How strict or lenient each annotator tended to be")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.35)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_annotator_tendencies.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Fig 3: Human–human agreement by question type ---
    rows = []
    for var, label in [
        ("register_markedness", "“Over-formal?”"),
        ("substitution_naturalness", "“Natural simpler word?”"),
        ("context_label", "Type of situation (e.g. informal / neutral)"),
    ]:
        subk = kappa[kappa["variable"] == var]
        rows.append({"label": label, "mean_agreement_pct": subk["percent_agreement"].mean()})
    dfm = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colors = ["#2A9D8F", "#E9C46A", "#264653"]
    bars = ax.barh(dfm["label"], dfm["mean_agreement_pct"], color=colors, height=0.55)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Average “both said the same” rate across all pairs of annotators (%)")
    ax.set_title("Do humans agree with each other? (first 10 shared examples)")
    for b, v in zip(bars, dfm["mean_agreement_pct"]):
        ax.text(v + 1.5, b.get_y() + b.get_height() / 2, f"{v:.0f}%", va="center", fontsize=10)
    ax.grid(axis="x", alpha=0.35)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig3_humans_agree_with_each_other.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Fig 4: Pooled essay topics (human-labeled rows only) ---
    top_n = min(12, len(topics_pool))
    tp = topics_pool.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * top_n + 1)))
    y_pos = np.arange(len(tp))
    ax.barh(y_pos, tp["n_rows_pooled_human_labels"], color="#457B9D")
    ax.set_yticks(y_pos)
    short = tp["essay_topic"].astype(str).str.slice(0, 55)
    ax.set_yticklabels(short, fontsize=9)
    ax.set_xlabel("Number of labeled rows (all annotators combined)")
    ax.set_title("Most common learner essay tasks in human-labeled rows")
    for i, (n, p) in enumerate(zip(tp["n_rows_pooled_human_labels"], tp["pct_of_pooled"])):
        ax.text(n + 0.3, i, f"{int(n)} ({p}%)", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.35)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig4_essay_topics_pooled.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Fig 5: Typical topic per annotator (share of their labels) ---
    ts = topics_sum[topics_sum["n_human_labeled_rows"] > 0].copy()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    y = np.arange(len(ts))
    ax.barh(y, ts["pct_in_most_common_topic"], color="#8D99AE")
    ax.set_yticks(y)
    ax.set_yticklabels(ts["annotator"])
    ax.set_xlabel("Share of that person’s labels (%)")
    ax.set_title("How much of each annotator’s work is their single most common essay task?")
    for i, (_, r) in enumerate(ts.iterrows()):
        lbl = str(r["most_common_essay_topic"])[:40] + (
            "…" if len(str(r["most_common_essay_topic"])) > 40 else ""
        )
        ax.text(r["pct_in_most_common_topic"] + 1, i, lbl, va="center", fontsize=8)
    ax.set_xlim(0, min(110, ts["pct_in_most_common_topic"].max() + 45))
    ax.grid(axis="x", alpha=0.35)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig5_typical_topic_per_annotator.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Fig 6: CEFR bands per annotator (stacked %) ---
    if not cefr_counts.empty:
        pivot = cefr_counts.pivot_table(
            index="annotator", columns="cefr_band", values="n_rows", fill_value=0
        )
        row_sums = pivot.sum(axis=1)
        pct = pivot.div(row_sums, axis=0) * 100
        pct = pct[sorted(pct.columns, key=str)]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        pct.plot(kind="barh", stacked=True, ax=ax, colormap="viridis", legend=True, width=0.7)
        ax.set_xlabel("Share of human-labeled rows (%)")
        ax.set_title("CEFR level of learner texts each annotator labeled")
        ax.legend(title="CEFR", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        ax.grid(axis="x", alpha=0.25)
        plt.tight_layout()
        fig.savefig(FIG_DIR / "fig6_cefr_bands_by_annotator.png", dpi=150, bbox_inches="tight")
        plt.close()

    # --- Fig 7: Pairwise Cohen's kappa — register (first 10) ---
    mat, mat_names = _pairwise_kappa_matrix(kappa, "register_markedness")
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(mat, vmin=-0.2, vmax=1.0, cmap="RdYlGn")
    ax.set_xticks(range(len(mat_names)))
    ax.set_yticks(range(len(mat_names)))
    ax.set_xticklabels(mat_names, rotation=45, ha="right")
    ax.set_yticklabels(mat_names)
    for i in range(len(mat_names)):
        for j in range(len(mat_names)):
            val = mat[i, j]
            txt = "—" if np.isnan(val) else (f"{val:.2f}" if val < 1 else "1.0")
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="black")
    ax.set_title("Pairwise agreement (κ) — “over-formal?”\n(first 10 shared examples)")
    fig.colorbar(im, ax=ax, shrink=0.6, label="Cohen's κ")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig7_kappa_register_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Fig 8: Pooled all judgments vs AI ---
    pr = pooled.iloc[0]
    fig, ax = plt.subplots(figsize=(5, 4))
    cats = ["“Over-formal?”\nmatch AI", "“Natural word?”\nmatch AI"]
    vals = [pr["accuracy_register"] * 100, pr["accuracy_substitution"] * 100]
    cols = ["#2E86AB", "#E94F37"]
    ax.bar(cats, vals, color=cols)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Match rate (%)")
    ax.set_title("All annotators pooled vs AI\n(same sentence can count more than once)")
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f"{v:.0f}%", ha="center", fontweight="bold")
    ax.grid(axis="y", alpha=0.35)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig8_pooled_vs_ai.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Fig 9: Scatter — strict on register vs strict on substitution ---
    hum2 = hum.copy()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    xr = hum2["mean_register_markedness_human"] * 100
    yr = hum2["mean_substitution_naturalness_human"] * 100
    ax.scatter(xr, yr, s=120, c="#E63946", edgecolors="white", zorder=3)
    for _, r in hum2.iterrows():
        ax.annotate(
            r["annotator"],
            (r["mean_register_markedness_human"] * 100, r["mean_substitution_naturalness_human"] * 100),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
        )
    ax.set_xlabel("Avg “over-formal?” yes rate (%)")
    ax.set_ylabel("Avg “natural simpler word?” yes rate (%)")
    ax.set_title("Annotator profiles (human-labeled rows only)")
    ax.set_xlim(65, 95)
    ax.set_ylim(35, 90)
    ax.grid(alpha=0.35)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig9_annotator_profile_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Wrote figures under {FIG_DIR}")


if __name__ == "__main__":
    main()
