from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_top_overformal(kind: str) -> None:
    """
    Plot top lemmas by overformal_count for 'small' or 'big' preannotated metrics.
    Saves PNGs into metrics/.
    """
    project_root = Path(__file__).resolve().parents[1]
    metrics_dir = project_root / "metrics"
    csv_path = metrics_dir / f"preannotated_metrics_{kind}.csv"
    if not csv_path.exists():
        print(f"{csv_path} not found, skipping top_overformal plot for {kind}.")
        return

    df = pd.read_csv(csv_path)
    if "overformal_count" not in df.columns:
        print(f"'overformal_count' not in {csv_path}, skipping.")
        return

    # Take top 10 by overformal_count
    top = df.sort_values("overformal_count", ascending=False).head(10)

    plt.figure(figsize=(10, 5))
    plt.bar(top["target_word"], top["overformal_count"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Over-formal count")
    plt.title(f"Top lemmas by over-formal count ({kind})")
    plt.tight_layout()

    out_png = metrics_dir / f"plot_preannotated_{kind}_top_overformal.png"
    plt.savefig(out_png)
    plt.close()
    print(f"Wrote {out_png}")


def plot_register_labels(kind: str) -> None:
    """
    Plot overall register_label distribution for 'small' or 'big' preannotated CSV.
    """
    project_root = Path(__file__).resolve().parents[1]
    pre_path = project_root / "approach_a" / "output" / f"preannotated_{kind}.csv"
    if not pre_path.exists():
        print(f"{pre_path} not found, skipping register_label plot for {kind}.")
        return

    df = pd.read_csv(pre_path)
    if "register_label" not in df.columns:
        print(f"'register_label' not in {pre_path}, skipping.")
        return

    counts = df["register_label"].astype(str).value_counts()

    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title(f"Register label distribution ({kind})")
    plt.tight_layout()

    metrics_dir = project_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_png = metrics_dir / f"plot_preannotated_{kind}_register_labels.png"
    plt.savefig(out_png)
    plt.close()
    print(f"Wrote {out_png}")


def main() -> None:
    # Visualise both small and big if available
    for kind in ("small", "big"):
        plot_top_overformal(kind)
        plot_register_labels(kind)


if __name__ == "__main__":
    main()

