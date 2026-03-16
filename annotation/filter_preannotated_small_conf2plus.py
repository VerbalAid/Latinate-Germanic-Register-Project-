from pathlib import Path

import pandas as pd


def main() -> None:
    """
    Build a SMALL annotation set with an even number of examples per lexicon word.

    For each latinate target in `annotation/lexicon/lexicon/Small_seed_lexicon.csv`,
    select up to N examples (default 2) from the SMALL pre-annotated dataset,
    preferring "good" LLM cases but backfilling if needed so the output is as even
    as possible.

    Input:
      - approach_a/output/preannotated_small.csv (preferred) OR
        Filtering_LLM_Filtering/output/preannotated_small.csv (fallback)
      - annotation/lexicon/lexicon/Small_seed_lexicon.csv

    Output:
      - annotation/preannotated_small_conf2plus.csv
    """
    project_root = Path(__file__).resolve().parents[1]
    pre_path = project_root / "approach_a" / "output" / "preannotated_small.csv"
    if not pre_path.exists():
        alt_path = (
            project_root
            / "Filtering_LLM_Filtering"
            / "output"
            / "preannotated_small.csv"
        )
        if alt_path.exists():
            pre_path = alt_path
        else:
            raise FileNotFoundError(
                "Could not find preannotated_small.csv at either:\n"
                f"  - {project_root / 'approach_a' / 'output' / 'preannotated_small.csv'}\n"
                f"  - {alt_path}\n"
            )
    out_path = project_root / "annotation" / "preannotated_small_conf2plus.csv"

    df = pd.read_csv(pre_path)

    lex_path = (
        project_root
        / "annotation"
        / "lexicon"
        / "lexicon"
        / "Small_seed_lexicon.csv"
    )
    lex = pd.read_csv(lex_path)
    if "latinate" not in lex.columns:
        raise ValueError(f"Expected 'latinate' column in {lex_path}")
    lex_targets = (
        lex["latinate"].astype(str).str.strip().str.lower().replace("nan", "")
    )
    lex_targets = [w for w in lex_targets.tolist() if w]

    df["register_markedness_llm"] = pd.to_numeric(
        df.get("register_markedness_llm", 0), errors="coerce"
    ).fillna(0).astype(int)
    df["confidence_llm"] = pd.to_numeric(
        df.get("confidence_llm", 1), errors="coerce"
    ).fillna(1).astype(int)

    # Normalise key text columns for matching
    df["target_word"] = df.get("target_word", "").astype(str).str.strip().str.lower()

    # Take up to N examples per lexicon target_word, with backfill tiers
    N_PER_LEMMA = 2
    tiers = [
        ("marked_conf2plus", (df["register_markedness_llm"] == 1) & (df["confidence_llm"] >= 2)),
        ("marked_anyconf", (df["register_markedness_llm"] == 1)),
        ("any_conf2plus", (df["confidence_llm"] >= 2)),
        ("any", pd.Series([True] * len(df), index=df.index)),
    ]

    picked_rows = []
    for lemma in lex_targets:
        sub = df[df["target_word"] == lemma].copy()
        if sub.empty:
            continue
        chosen = []
        chosen_ids = set()
        for tier_name, tier_mask in tiers:
            candidates = sub[tier_mask.loc[sub.index]].copy()
            if candidates.empty:
                continue
            # Prefer higher confidence first inside a tier
            candidates = candidates.sort_values(["confidence_llm"], ascending=False)
            for _, row in candidates.iterrows():
                if len(chosen) >= N_PER_LEMMA:
                    break
                # (sentence_id, target_word) is a good-enough uniqueness key here
                key = (row.get("sentence_id"), row.get("target_word"))
                if key in chosen_ids:
                    continue
                r = row.copy()
                r["selection_tier"] = tier_name
                chosen.append(r)
                chosen_ids.add(key)
            if len(chosen) >= N_PER_LEMMA:
                break
        picked_rows.extend(chosen)

    sampled = pd.DataFrame(picked_rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(out_path, index=False)

    print(
        f"Selected {sampled.shape[0]} rows out of {df.shape[0]} "
        f"into {out_path}"
    )


if __name__ == "__main__":
    main()

