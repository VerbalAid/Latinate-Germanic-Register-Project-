from pathlib import Path

import pandas as pd


def main() -> None:
    """
    Compute useful summary metrics for the SMALL preannotated corpus.

    Input:
      - approach_a/output/preannotated_small.csv
    Output:
      - metrics/preannotated_metrics_small.csv

    Per target_word (latinate lemma) we compute:
      - total_count: how many candidate rows
      - overformal_count: how many with register_markedness_llm == 1
      - overformal_ratio: overformal_count / total_count
      - mean_confidence: mean confidence_llm over all rows
      - mean_confidence_overformal: mean confidence_llm where register_markedness_llm == 1
      - top_register_label: most frequent register_label for that lemma
    """
    project_root = Path(__file__).resolve().parents[1]
    pre_path = project_root / "approach_a" / "output" / "preannotated_small.csv"
    if not pre_path.exists():
        alt = (
            project_root
            / "Filtering_LLM_Filtering"
            / "output"
            / "preannotated_small.csv"
        )
        if alt.exists():
            pre_path = alt
        else:
            raise FileNotFoundError(
                f"preannotated_small.csv not found at {pre_path} or {alt}"
            )

    df = pd.read_csv(pre_path)

    # Normalise numeric fields
    df["register_markedness_llm"] = pd.to_numeric(
        df.get("register_markedness_llm", 0), errors="coerce"
    ).fillna(0).astype(int)
    df["confidence_llm"] = pd.to_numeric(
        df.get("confidence_llm", 1), errors="coerce"
    ).fillna(1).astype(int)

    # Basic counts per lemma
    total = (
        df.groupby("target_word")
        .size()
        .reset_index(name="total_count")
    )

    overformal = (
        df[df["register_markedness_llm"] == 1]
        .groupby("target_word")
        .size()
        .reset_index(name="overformal_count")
    )

    metrics = total.merge(overformal, on="target_word", how="left")
    metrics["overformal_count"] = metrics["overformal_count"].fillna(0).astype(int)
    metrics["overformal_ratio"] = metrics["overformal_count"] / metrics["total_count"]

    # Mean confidence overall and for overformal cases only
    mean_conf_all = (
        df.groupby("target_word")["confidence_llm"]
        .mean()
        .reset_index(name="mean_confidence")
    )

    mean_conf_over = (
        df[df["register_markedness_llm"] == 1]
        .groupby("target_word")["confidence_llm"]
        .mean()
        .reset_index(name="mean_confidence_overformal")
    )

    metrics = metrics.merge(mean_conf_all, on="target_word", how="left")
    metrics = metrics.merge(mean_conf_over, on="target_word", how="left")

    # Most frequent register_label per lemma
    top_labels = []
    for lemma, sub in df.groupby("target_word"):
        if "register_label" in sub.columns:
            label = sub["register_label"].astype(str).value_counts().idxmax()
        else:
            label = ""
        top_labels.append({"target_word": lemma, "top_register_label": label})
    top_labels_df = pd.DataFrame(top_labels)

    metrics = metrics.merge(top_labels_df, on="target_word", how="left")

    # Write metrics
    metrics_dir = project_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / "preannotated_metrics_small.csv"
    metrics.sort_values("overformal_ratio", ascending=False).to_csv(
        out_path, index=False
    )

    # Also print a short view
    print(metrics.sort_values("overformal_ratio", ascending=False))


if __name__ == "__main__":
    main()

