from pathlib import Path

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    # Path to the preannotated file built from candidates_small.csv
    pre_path = project_root / "approach_a" / "output" / "preannotated.csv"
    df = pd.read_csv(pre_path)

    # Make sure the score column is numeric
    df["register_markedness_llm"] = pd.to_numeric(
        df["register_markedness_llm"], errors="coerce"
    ).fillna(0.0)

    # Choose a threshold for "clear" over-formal; adjust as needed
    THRESHOLD = 3.0

    clear = df[df["register_markedness_llm"] >= THRESHOLD]

    stats = (
        clear.groupby("target_word")
        .size()
        .reset_index(name="clear_count")
        .sort_values("clear_count", ascending=False)
    )

    # Write metrics to metrics/ so they are easy to inspect later.
    metrics_dir = project_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / "lemma_frequencies_preannotated_small.csv"
    stats.to_csv(out_path, index=False)

    # Also print to stdout for a quick look.
    print(stats)


if __name__ == "__main__":
    main()


