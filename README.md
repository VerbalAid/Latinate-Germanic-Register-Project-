
# Latinate vs Germanic in Learner English

This project builds a small language resource on over‑formal Latinate verb choices in learner English, based on the EFCAMDAT corpus.

## Layout

- `data/` – raw corpus files (for example `raw/efcamdat_full.csv`).
- `annotation/` – hand‑built lexica and human‑annotation helpers.
  - `lexicon/lexicon/Small_seed_lexicon.csv` – 20 Germanic–Latinate pairs with register labels.
  - `lexicon/lexicon/Big_seed_lexicon.csv` – larger seed lexicon (optional).
- `approach_a/` – main pipeline code.
- `analytics/` – analysis scripts.
- `metrics/` – CSV and plot outputs from analytics.

There is also a `Filtering_LLM_Filtering/` folder which contains an earlier, experimental version of the small‑lexicon pipeline. You do not need it for the main workflow; it is kept only as an archive of earlier experiments.

## Quick start (small lexicon only)

From the project root:

```bash
cd ~/Desktop/Language\ Resources/Project

# 1. Build candidates (small 20‑pair lexicon)
python approach_a/step1_lexicon_filter_small.py

# 2. Run Ollama pre‑annotation (with mistral:7b running)
python approach_a/step2_ollama_preannotate_small.py

# 3. Basic analytics and plots
python analytics/preannotated_metrics_small.py
python analytics/visualize_metrics.py
```

## Pipeline

### 1. Build candidates

**Small 20‑pair lexicon:**

```bash
python approach_a/step1_lexicon_filter_small.py
```

This:

- reads `data/raw/efcamdat_full.csv`
- keeps French/Spanish L1 learners with CEFR A1–C1
- samples up to `N_PER_CELL` essays per (L1, CEFR) cell
- finds sentences containing Latinate lemmas from `Small_seed_lexicon.csv`
- writes one row per (sentence, lemma) to `approach_a/candidates_small.csv`

**Big lexicon (optional):**

```bash
python approach_a/step1_lexicon_filter_big.py
```

Same idea, but using `Big_seed_lexicon.csv` and writing `candidates_big.csv`.

### 2. LLM pre‑annotation (Ollama)

First start Ollama in another terminal and ensure the model is installed:

```bash
ollama serve
ollama pull mistral:7b
```

Then run:

```bash
python approach_a/step2_ollama_preannotate_small.py
# optional:
python approach_a/step2_ollama_preannotate_big.py
```

Each script:

- reads `candidates_*.csv`
- calls `mistral:7b` via the Ollama HTTP API
- adds:
  - `register_label` (informal / neutral essay / academic / news / ambiguous)
  - `register_markedness_llm` (0/1, over‑formal or not)
  - `substitution_natural_llm` (0/1)
  - `simpler_alternative_llm` (string)
  - `confidence_llm` (1–3)
- writes `preannotated_small.csv` or `preannotated_big.csv` in `approach_a/`.

### 3. Analytics and plots

Run:

```bash
# lemma frequencies in the candidate sets
python analytics/lemma_frequencies_candidates_small.py
python analytics/lemma_frequencies_candidates_big.py

# detailed metrics from pre‑annotated files
python analytics/preannotated_metrics_small.py
python analytics/preannotated_metrics_big.py

# bar charts / visual summaries
python analytics/visualize_metrics.py
```

These scripts read the relevant `candidates_*.csv` and `preannotated_*.csv` files and write CSVs and PNGs into `metrics/` (for example lemma frequencies, over‑formal ratios, and register‑label distributions).

### 4. Human annotation files

To prepare CSVs for manual annotation (LLM + human columns side by side):

```bash
python annotation/make_small_to_human_annotate.py
python annotation/make_big_to_human_annotate.py
```

These read `approach_a/output/preannotated_small.csv` / `_big.csv` and write:

- `annotation/Small_to_Human_Annotate.csv`
- `annotation/Big_to_Human_Annotate.csv`

## Key files

- `approach_a/step1_lexicon_filter_small.py`  
  Filters the main corpus using the 20‑pair small lexicon and produces `candidates_small.csv`.

- `approach_a/step1_lexicon_filter_big.py`  
  Same as above but for the larger lexicon and produces `candidates_big.csv`.

- `approach_a/step2_ollama_preannotate_small.py`  
  Calls Ollama (Mistral 7B) to pre‑annotate each candidate row and writes `preannotated_small.csv`.

- `approach_a/step2_ollama_preannotate_big.py`  
  As above, but for the big lexicon and writes `preannotated_big.csv`.

- `approach_a/filter_overformal.py`  
  Helper script to filter pre‑annotations by `register_markedness_llm` (keeps over‑formal cases).

- `analytics/lemma_frequencies_candidates_small.py` / `analytics/lemma_frequencies_candidates_big.py`  
  Count how often each Latinate lemma appears in the candidate sets and save CSVs in `metrics/`.

- `analytics/preannotated_metrics_small.py` / `analytics/preannotated_metrics_big.py`  
  Compute per‑lemma totals, over‑formal counts and ratios, mean confidence, and the most common register label and save CSVs in `metrics/`.

- `analytics/visualize_metrics.py`  
  Produces bar charts (top over‑formal lemmas and register‑label distributions) and saves them as PNGs in `metrics/`.

