"""
step1_lexicon_filter_big.py

Very simple script:
- Reads the full learner corpus: data/raw/efcamdat_full.csv
- Uses the BIG lexicon to find latinate target words
- Keeps only Spanish / French learners in A1–C1
- Samples up to N_PER_CELL essays per (L1, CEFR) cell
- Writes candidates to approach_a/output/candidates_big.csv
"""

from pathlib import Path

import pandas as pd


# --- Paths and basic settings ------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

RAW_CSV = PROJECT_ROOT / "data" / "raw" / "efcamdat_full.csv"

# ADJUST THIS PATH IF YOUR BIG LEXICON LIVES SOMEWHERE ELSE.
BIG_LEXICON_CSV = PROJECT_ROOT / "annotation" / "lexicon" / "lexicon" / "Big_seed_lexicon.csv"

OUT_CSV = OUTPUT_DIR / "candidates_big.csv"

# Target learner L1s and CEFR levels
TARGET_L1S = ("Spanish", "French")
TARGET_CEFR = ("A1", "A2", "B1", "B2", "C1")

# Maximum number of essays per (L1, CEFR) cell              -play with this to get more matches 
N_PER_CELL = 10000


def load_big_lexicon(lexicon_path: Path):
    """Read the big lexicon and return:
    - surface_forms: set of surface word forms
    - lemmas: set of base lemmas
    - surface_to_lemma: mapping surface_form -> lemma
    """
    if not lexicon_path.exists():
        raise FileNotFoundError(f"Big lexicon not found: {lexicon_path}")

    df = pd.read_csv(lexicon_path)
    if "latinate" not in df.columns:
        raise ValueError(f"Lexicon must have a 'latinate' column: {lexicon_path}")

    lemmas = (
        df["latinate"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    lemmas = {lemma for lemma in lemmas if lemma and lemma != "nan"}

    surface_forms = set()
    surface_to_lemma = {}

    for lemma in lemmas:
        # Very simple inflection patterns
        base_forms = [lemma, lemma + "s", lemma + "ed", lemma + "ing"]
        for surface in base_forms:
            surface_forms.add(surface)
            surface_to_lemma[surface] = lemma

        # -ise / -ize endings
        if lemma.endswith("ise"):
            for surface in (lemma + "d", lemma + "s"):
                surface_forms.add(surface)
                surface_to_lemma[surface] = lemma
        if lemma.endswith("ize"):
            for surface in (lemma + "d", lemma + "s"):
                surface_forms.add(surface)
                surface_to_lemma[surface] = lemma

    return surface_forms, lemmas, surface_to_lemma


def simple_tokenise(text: str):
    if not isinstance(text, str):
        return []
    cleaned = "".join(c.lower() if c.isalpha() else " " for c in text)
    return cleaned.split()


def filter_es_fr_cefr(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["l1"].isin(TARGET_L1S) & df["cefr"].isin(TARGET_CEFR)].copy()
    df = df[df["text"].astype(str).str.strip().ne("")]
    return df


def balanced_sample(df: pd.DataFrame) -> pd.DataFrame:
    groups = df.groupby(["l1", "cefr"])
    samples = []
    for (_l1, _cefr), grp in groups:
        n = min(N_PER_CELL, len(grp))
        if n > 0:
            samples.append(grp.sample(n=n, random_state=42))
    if not samples:
        raise ValueError("No data after filtering – check L1/CEFR columns.")
    out = pd.concat(samples, ignore_index=True)
    if "sentence_id" not in out.columns:
        out.insert(0, "sentence_id", range(1, len(out) + 1))
    return out


def expand_to_candidates(df: pd.DataFrame, surface_forms, surface_to_lemma):
    records = []
    for _, row in df.iterrows():
        text = str(row["text"])
        tokens = simple_tokenise(text)
        seen_lemmas = set()
        for t in tokens:
            if t in surface_forms:
                seen_lemmas.add(surface_to_lemma[t])
        for lemma in sorted(seen_lemmas):
            records.append(
                {
                    "sentence_id": row["sentence_id"],
                    "l1": row["l1"],
                    "cefr": row["cefr"],
                    "cefr_numeric": row["cefr_numeric"],
                    "topic": row["topic"],
                    "grade": row["grade"],
                    "wordcount": row["wordcount"],
                    "nationality": row["nationality"],
                    "text": row["text"],
                    "target_word": lemma,
                }
            )
    return pd.DataFrame.from_records(records)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    surface_forms, lemmas, surface_to_lemma = load_big_lexicon(BIG_LEXICON_CSV)
    print(f"Big lexicon: {len(lemmas)} lemmas, {len(surface_forms)} surface forms.")

    df = pd.read_csv(RAW_CSV)
    df = filter_es_fr_cefr(df)
    df = balanced_sample(df)
    print(f"Sample shape: {df.shape}")

    candidates_df = expand_to_candidates(df, surface_forms, surface_to_lemma)
    print(f"Total candidates: {len(candidates_df)}")

    candidates_df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(candidates_df)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()

