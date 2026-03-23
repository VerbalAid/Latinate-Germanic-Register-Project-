"""
Plots for preannotated_small.csv profile metrics.

Run after:
  python analytics/preannotated_small_profile.py
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
FIG = PROJECT / "docs" / "figures"


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
        }
    )

    overall = pd.read_csv(METRICS / "preannotated_small_overall.csv").iloc[0]
    reg = pd.read_csv(METRICS / "preannotated_small_register_label_dist.csv")
    conf = pd.read_csv(METRICS / "preannotated_small_confidence_dist.csv")
    lemma = pd.read_csv(METRICS / "preannotated_small_lemma_stats.csv")
    lex_tok = pd.read_csv(METRICS / "preannotated_small_lexicon_token_counts.csv")
    l1_tab = pd.read_csv(METRICS / "preannotated_small_l1_by_overformal.csv")
    cefr_tab = pd.read_csv(METRICS / "preannotated_small_cefr_by_overformal.csv")

    n = int(overall["n_rows"])
    pct_m = float(overall["pct_over_formal_llm"])
    n_marked = int(overall["n_register_marked_llm_1"])
    n_unmarked = int(overall["n_register_marked_llm_0"])

    # --- Fig A: Over-formal share (pie) ---
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    def _pie_autopct(pct):
        absolute = int(round(pct / 100.0 * n))
        return f"{absolute}\n({pct:.1f}%)"

    ax.pie(
        [n_marked, n_unmarked],
        labels=[
            f"Marked over-formal\n({pct_m:.1f}% of rows)",
            f"Not marked\n({100 - pct_m:.1f}% of rows)",
        ],
        autopct=_pie_autopct,
        colors=["#E07A5F", "#81B29A"],
        startangle=90,
        textprops={"fontsize": 10},
    )
    ax.set_title(
        f'LLM: register marked (over-formal)\npreannotated_small.csv — N = {n:,} rows'
    )
    plt.tight_layout()
    fig.savefig(FIG / "preannot_profile_overformal_pie.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Fig B: register_label bars ---
    if not reg.empty:
        r = reg.sort_values("count", ascending=True).tail(12)
        fig, ax = plt.subplots(figsize=(8, 4.8))
        y = np.arange(len(r))
        ax.barh(y, r["count"], color="#3D5A80")
        ax.set_yticks(y)
        ax.set_yticklabels(
            r["register_label"].astype(str).str.slice(0, 40), fontsize=9
        )
        ax.set_xlabel("Number of rows")
        ax.set_title("Distribution of context register_label (LLM)")
        for i, (c, p) in enumerate(zip(r["count"], r["pct_of_rows"])):
            ax.text(c + n * 0.005, i, f"{p:.1f}%", va="center", fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        fig.savefig(
            FIG / "preannot_profile_register_label_bars.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

    # --- Fig C: confidence bars ---
    fig, ax = plt.subplots(figsize=(5.5, 4))
    x = conf["confidence_llm"].astype(int).tolist()
    h = conf["count"].tolist()
    colors = ["#8D99AE", "#EDAE49", "#44AF69"]
    bar_cols = [colors[min(i, 2)] for i in range(len(x))]
    ax.bar([str(c) for c in x], h, color=bar_cols[: len(x)])
    ax.set_xlabel("confidence_llm (1=low … 3=high)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of LLM confidence")
    for i, (xi, hi, p) in enumerate(zip(x, h, conf["pct_of_rows"])):
        ax.text(i, hi + n * 0.01, f"{p:.1f}%", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG / "preannot_profile_confidence_bars.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Fig D: lemmas — most flagged (count) ---
    top_n = min(15, len(lemma))
    lf = lemma.nlargest(top_n, "n_flagged_overformal").iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    y = np.arange(len(lf))
    ax.barh(y, lf["n_flagged_overformal"], color="#9B2335", label="Flagged count")
    ax.set_yticks(y)
    ax.set_yticklabels(lf["target_word"])
    ax.set_xlabel("Rows flagged as over-formal (register_markedness_llm = 1)")
    ax.set_title("Lemmas with the most over-formal flags (raw counts)")
    ax2 = ax.twiny()
    ax2.scatter(
        lf["pct_flagged"],
        y,
        color="#2B2D42",
        s=45,
        zorder=4,
        label="% of that lemma’s rows",
    )
    ax2.set_xlabel("Flag rate for that lemma (%)", color="#2B2D42")
    ax2.tick_params(axis="x", colors="#2B2D42")
    ax.grid(axis="x", alpha=0.3)
    fig.legend(loc="lower right", bbox_to_anchor=(0.98, 0.02), fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG / "preannot_profile_lemmas_flagged.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Fig E: lemmas — highest mean confidence (min 5 rows) ---
    sub = lemma[lemma["n_rows"] >= 5].nlargest(min(15, len(lemma)), "mean_confidence_llm")
    sub = sub.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    y = np.arange(len(sub))
    ax.barh(y, sub["mean_confidence_llm"], color="#5C4D7D")
    ax.set_yticks(y)
    ax.set_yticklabels(sub["target_word"])
    ax.set_xlim(1, 3.15)
    ax.set_xlabel("Mean confidence_llm (1–3)")
    ax.set_title("Lemmas with highest mean LLM confidence\n(lemmas with ≥5 rows only)")
    for i, nr in enumerate(sub["n_rows"]):
        ax.text(
            min(sub["mean_confidence_llm"].iloc[i] + 0.04, 3.05),
            i,
            f"n={nr}",
            va="center",
            fontsize=8,
            color="#444",
        )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(
        FIG / "preannot_profile_lemmas_mean_confidence.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    # --- Fig F: Small lexicon — distribution of row counts per latinate token ---
    lt = lex_tok.sort_values("n_rows_in_preannotated", ascending=True).copy()
    lt["label"] = lt.apply(
        lambda r: f"{r['latinate']} ({r['germanic_pair']})"
        if pd.notna(r["germanic_pair"]) and str(r["germanic_pair"]).strip()
        else str(r["latinate"]),
        axis=1,
    )
    y = np.arange(len(lt))
    colors = np.where(lt["n_rows_in_preannotated"].values > 0, "#457B9D", "#B8B8B8")
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.barh(y, lt["n_rows_in_preannotated"], color=colors, height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(lt["label"], fontsize=9)
    ax.set_xlabel("Number of preannotated rows (essay × target hits)")
    ax.set_title(
        "Small lexicon: how often each latinate type appears\nin preannotated_small.csv (gray = 0 rows)"
    )
    xmax = max(lt["n_rows_in_preannotated"].max(), 1) * 1.08
    ax.set_xlim(0, xmax)
    for i, (cnt, pct) in enumerate(
        zip(lt["n_rows_in_preannotated"], lt["pct_of_preannotated_rows"])
    ):
        if cnt > 0:
            ax.text(
                cnt + xmax * 0.01,
                i,
                f"{int(cnt)} ({pct:.2f}%)",
                va="center",
                fontsize=8,
                color="#333",
            )
        else:
            ax.text(xmax * 0.01, i, "0", va="center", fontsize=8, color="#666")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(
        FIG / "preannot_profile_lexicon_token_distribution.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # --- Fig G & H: L1 and CEFR vs over-formal (100% stacked) ---
    def _stacked_overformal_fig(
        tab: pd.DataFrame,
        key_col: str,
        title: str,
        fname: str,
        figsize: tuple[float, float],
    ) -> None:
        if tab.empty or key_col not in tab.columns:
            return
        n0 = tab["n_not_overformal_llm_0"].to_numpy(dtype=float)
        n1 = tab["n_overformal_llm_1"].to_numpy(dtype=float)
        tot = n0 + n1
        with np.errstate(divide="ignore", invalid="ignore"):
            p0 = np.where(tot > 0, 100.0 * n0 / tot, 0.0)
            p1 = np.where(tot > 0, 100.0 * n1 / tot, 0.0)
        x = np.arange(len(tab))
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(x, p0, color="#81B29A", label="LLM: not over-formal (0)", width=0.72)
        ax.bar(x, p1, bottom=p0, color="#E07A5F", label="LLM: over-formal (1)", width=0.72)
        ax.set_ylim(0, 100)
        ax.set_ylabel("% of rows in that group")
        ax.set_title(title)
        labs = [
            f"{lab}\n(n={int(t)})"
            for lab, t in zip(tab[key_col].astype(str), tot)
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(labs, fontsize=10)
        ax.legend(loc="upper right", framealpha=0.95)
        ax.grid(axis="y", alpha=0.3)
        for i in x:
            if p1[i] > 8:
                ax.text(
                    i,
                    p0[i] + p1[i] / 2,
                    f"{p1[i]:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white",
                    fontweight="bold",
                )
            elif p0[i] > 8:
                ax.text(
                    i,
                    p0[i] / 2,
                    f"{p0[i]:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#1d3557",
                )
        plt.tight_layout()
        fig.savefig(FIG / fname, dpi=150, bbox_inches="tight")
        plt.close()

    _stacked_overformal_fig(
        l1_tab,
        "l1",
        "Share of rows marked over-formal by learner L1\n(100% stacked; LLM register_markedness_llm)",
        "preannot_profile_l1_overformal_stacked.png",
        (6.5, 4.6),
    )
    _stacked_overformal_fig(
        cefr_tab,
        "cefr",
        "Share of rows marked over-formal by CEFR band\n(100% stacked; LLM register_markedness_llm)",
        "preannot_profile_cefr_overformal_stacked.png",
        (7.5, 4.6),
    )

    print(f"Wrote plots under {FIG}")


if __name__ == "__main__":
    main()
