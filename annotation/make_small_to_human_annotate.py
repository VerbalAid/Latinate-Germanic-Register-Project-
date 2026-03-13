from pathlib import Path

import pandas as pd


def main() -> None:
    """
    Create a filtered CSV of "good" SMALL-lexicon examples for human annotation.

    Input:
      - approach_a/output/preannotated_small.csv

    Output:
      - annotation/Small_to_Human_Annotate.csv

    The output file has:
      - One header row with column names and short hints in brackets
      - One row per selected example
    """
    project_root = Path(__file__).resolve().parents[1]
    pre_path = project_root / "approach_a" / "output" / "preannotated_small.csv"
    out_path = project_root / "annotation" / "Small_to_Human_Annotate.csv"

    df = pd.read_csv(pre_path)

    # Normalise and coerce LLM fields
    df["register_markedness_llm"] = pd.to_numeric(
        df.get("register_markedness_llm", 0), errors="coerce"
    ).fillna(0).astype(int)
    df["substitution_natural_llm"] = pd.to_numeric(
        df.get("substitution_natural_llm", 0), errors="coerce"
    ).fillna(0).astype(int)
    df["confidence_llm"] = pd.to_numeric(
        df.get("confidence_llm", 1), errors="coerce"
    ).fillna(1).astype(int)

    # "Good example" filter:
    #   - LLM thinks the Latinate choice is register-marked (1)
    #   - LLM thinks the simpler alternative is natural (1)
    #   - LLM has at least medium confidence (>= 2)
    mask = (
        (df["register_markedness_llm"] == 1)
        & (df["substitution_natural_llm"] == 1)
        & (df["confidence_llm"] >= 2)
    )
    good = df.loc[mask].copy()

    if good.empty:
        print("No rows matched the 'good example' criteria.")
        return

    # Some metadata columns might not exist in the SMALL pipeline, or may exist
    # under slightly different names. Where possible, copy from the existing
    # lower‑case source columns; otherwise, create empty ones so humans can
    # fill them in.
    if "L1" not in good.columns:
        if "l1" in good.columns:
            good["L1"] = good["l1"]
        else:
            good["L1"] = ""
    if "CEFR" not in good.columns:
        if "cefr" in good.columns:
            good["CEFR"] = good["cefr"]
        else:
            good["CEFR"] = ""
    for meta_col in ["topic", "grade", "nationality"]:
        if meta_col not in good.columns:
            good[meta_col] = ""

    # Empty HUMAN columns for annotators to fill in
    good["human_register_markedness"] = ""
    good["human_substitution_naturalness"] = ""
    good["human_simpler_alternative"] = ""
    good["human_confidence"] = ""
    good["human_short_comment"] = ""
    good["human_annotator_id"] = ""

    cols = [
        "sentence_id",
        "target_word",
        "text",
        "L1",
        "CEFR",
        "topic",
        "grade",
        "nationality",
        "register_label",
        "register_markedness_llm",
        "substitution_natural_llm",
        "simpler_alternative_llm",
        "confidence_llm",
        "human_register_markedness",
        "human_substitution_naturalness",
        "human_simpler_alternative",
        "human_confidence",
        "human_short_comment",
        "human_annotator_id",
    ]

    missing = [c for c in cols if c not in good.columns]
    if missing:
        raise KeyError(f"Missing expected columns in preannotated_small.csv: {missing}")

    good = good[cols]

    # Prepare header with short hints in brackets (for humans)
    header_with_hints = [
        "sentence_id (do not edit)",
        "target_word (Latinate verb under discussion)",
        "text (learner sentence containing the target_word)",
        "L1 (learner first language)",
        "CEFR (A1/A2/B1/B2/C1 from corpus)",
        "topic (essay topic, if available)",
        "grade (overall task grade / score, if available)",
        "nationality (learner nationality, if available)",
        "register_label (LLM context label: informal / neutral essay / academic / news / ambiguous)",
        "register_markedness_llm (LLM: 0 = not over‑formal, 1 = over‑formal)",
        "substitution_natural_llm (LLM: 0 = alternative not clearly better, 1 = natural replacement)",
        "simpler_alternative_llm (LLM: Germanic / simpler verb it suggests)",
        "confidence_llm (LLM: 1–3 confidence about its own judgement)",
        "human_register_markedness (YOU: 0 = not over‑formal, 1 = clearly over‑formal)",
        "human_substitution_naturalness (YOU: 0 = alternative not clearly better, 1 = natural replacement)",
        "human_simpler_alternative (YOU: Germanic / simpler verb you would actually use)",
        "human_confidence (YOU: 0–3, how sure you are)",
        "human_short_comment (very short explanation in English, optional)",
        "human_annotator_id (your initials or code)",
    ]

    # Write header + data (no instruction lines, for clean CSV import)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(header_with_hints) + "\n")
        good.to_csv(f, index=False, header=False)

    print(f"Wrote filtered human-annotation file to {out_path}")


if __name__ == "__main__":
    main()

