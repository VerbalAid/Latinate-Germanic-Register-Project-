from pathlib import Path

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    # Path to the candidates file built from the small lexicon
    candidates_path = project_root / "approach_a" / "output" / "candidates_small.csv"
    df = pd.read_csv(candidates_path)

    freq = (
        df.groupby("target_word")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    # Write metrics to metrics/ so they are easy to inspect later.
    metrics_dir = project_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / "lemma_frequencies_candidates_small.csv"
    freq.to_csv(out_path, index=False)

    # Also print to stdout for a quick look.
    print(freq)


if __name__ == "__main__":
    main()


