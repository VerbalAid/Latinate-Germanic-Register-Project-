# Interpretation: group annotation agreement & descriptives

This document interprets the outputs from `analytics/group_annotation_agreement.py` (Cohen’s κ, human–LLM agreement, and summary statistics). Numbers match the last generated CSVs in `metrics/` (re-run the script if files change).

---

## 1. What was measured

| Analysis | Data | Meaning |
|----------|------|--------|
| **Pairwise Cohen’s κ** | First **10** items in `preannotated_small_conf2plus.csv` (same `sentence_id` + `target_word`) | How often **two human annotators** give the **same** label, corrected for chance. |
| **Human vs LLM** | All rows that **merge** annotator sheets with the reference file; only rows where **both** human and LLM fields are non-missing | Agreement (κ and raw **accuracy**) between each annotator and the **LLM columns in the reference CSV**. |
| **Pooled human vs LLM** | Stacks valid human–LLM pairs from **all four** annotators | Overall pattern when every judgment counts separately (**67** pairs per variable)—the same sentence can contribute up to four times. |
| **Descriptives** | Per-file and reference corpus | Means and dominant categories for **L1**, **nationality**, **CEFR**, human **register** / **substitution** / **confidence**. |

**κ scale (Landis & Koch, rough guide):**  
&lt; 0 poor · 0.00–0.20 slight · 0.21–0.40 fair · 0.41–0.60 moderate · 0.61–0.80 substantial · 0.81–1.00 almost perfect  

---

## 2. Inter-rater agreement (first 10 items)

### 2.1 Register markedness (binary)

- **Perfect agreement (κ = 1.0, 100%)** for: **Darragh–Josu**, **Darragh–molina**, **Josu–molina** (all on **n = 10** overlapping ratings).
- Whenever **Numidia** is in the pair, **n = 9** (one of the ten items lacks a usable human register score for that rater), and **κ ≈ 0.73** with **~89%** agreement—still **substantial** agreement by conventional labels.

**Takeaway:** On the shared calibration set, humans largely **agree on whether the latinate use is register-marked**, with Numidia slightly less aligned on one item or missing one label.

### 2.2 Substitution naturalness (binary)

Agreement is **weaker** than for register:

- **Darragh–Josu** and **Darragh–molina:** **κ ≈ 0.29** (~**60%** agreement)—only **slight** beyond chance.
- **Josu–molina:** **κ = 1.0** (100%)—they match perfectly on these ten items.
- Pairs involving **Numidia:** **κ** between **~0.40 and ~0.73**—**fair to substantial**, depending on the pair.

**Takeaway:** The **substitution naturalness** judgment is **harder** or **less consistently operationalized** than register markedness; some pairs align almost perfectly, others diverge on ~40% of items.

### 2.3 Context label (nominal)

- **Darragh–Josu:** **κ = 1.0** (100%).
- **Numidia–molina:** **κ = 1.0** (100%).
- **Darragh–molina** and **Josu–molina:** **κ ≈ 0.60** (~**80%** agreement)—**moderate**.
- Pairs with **Numidia** (vs Darragh/Josu): **κ ≈ 0.77** (~**89%** agreement)—**substantial**.

**Takeaway:** Context labels are **fairly stable** across most pairs; the largest spread is **molina vs Darragh/Josu** on this small set.

---

## 3. Human vs LLM (merged rows)

### 3.1 Register markedness vs `register_markedness_llm`

- **Cohen’s κ for register is not reported** (empty in CSV) because on merged rows the **LLM column is constant** (`llm_register_variance_on_merged` = **False**): the reference file codes **all** items as marked (**mean LLM = 1.0**).
- **Accuracy** (raw match rate) is still meaningful:
  - **Darragh ~71%**, **Numidia ~81%**, **Josu ~88%**, **molina ~88%** with the LLM’s all-`1` policy.

**Takeaway:** The LLM **never predicts “unmarked”** in this reference slice, so κ **cannot** summarize “beyond-chance” discrimination. Humans **disagree** with that all-marked stance on **~12–29%** of labeled rows (depending on annotator). For papers, report **accuracy** or **prevalence-adjusted** metrics after fixing or stratifying the LLM output.

### 3.2 Substitution naturalness vs `substitution_natural_llm`

- **κ is negative** for **Darragh, Josu, Numidia** (~**−0.31 to −0.33**) and **near zero** for **molina** (~**+0.01**).
- **Negative κ** means agreement is **worse than chance** given the marginal distributions (e.g. humans and LLM systematically **invert** or use different thresholds).
- **Accuracy** is **low**: **~35–47%** for most annotators (**Numidia ~44%**, **molina ~47%**).

**Takeaway:** **Substitution naturalness** is **not aligned** between humans and this LLM layer on the current definitions—treat LLM substitution flags as **weak** for this task unless you recalibrate or re-annotate with clearer guidelines.

### 3.3 Pooled (67 pairs per variable)

- **Register:** again **no κ** (LLM side still constant); **accuracy ~82%** pooled.
- **Substitution:** **κ ≈ −0.26**, **accuracy ~43%**.

**Takeaway:** Pooling **does not** fix the substitution mismatch; it confirms **systematic human–LLM disagreement** on that variable.

---

## 4. Descriptive analytics (who annotated what)

| Source | Rows in file | Human register/substitution filled | Mean human register (1 = marked) | Mean human substitution (1 = natural) | Mean confidence (human) | Mean CEFR numeric* |
|--------|--------------|-------------------------------------|-------------------------------------|----------------------------------------|-------------------------|-------------------|
| Darragh | 38 | 17 | **0.71** | **0.82** | ~2.94 | ~1.76 |
| Josu | 38 | 17 | **0.88** | **0.53** | ~2.65 | ~2.06 |
| Numidia | 38 | 17 | **0.82** | **0.82** | ~2.88 | ~2.18 |
| molina | 38 | 17 | **0.88** | **0.41** | ~2.76 | ~2.00 |
| LLM reference | 38 | (N/A) | — | — | — | **2.0** (corpus) |

\*CEFR numeric mean is computed on rows where human register is filled (annotator slice).

**L1 / nationality / CEFR band:** Annotated rows are predominantly **French L1**, nationality **`fr`**. Modal CEFR band in the filled subset is often **A1** (Darragh) or **A2** (others)—consistent with learner data, not a property of the annotator.

**Cross-annotator style signals:**

- **Darragh** rates **fewer** instances as register-marked (**0.71**) but is **lenient** on substitution naturalness (**0.82**).
- **Josu** and **molina** rate **more** register-marked (**0.88**); **molina** is **strictest** on substitution naturalness (**0.41** vs **0.82** for Darragh/Numidia).
- **Numidia** sits **between** on register (**0.82**) and matches Darragh on substitution leniency (**0.82**).

**LLM reference (corpus-level):** Mean **register_markedness_llm = 1.0** (always marked), mean **substitution_natural_llm ≈ 0.66**, mean **confidence_llm = 3**—so the model is **confident** and **uniform** on register in this file.

---

## 5. Limitations (short)

1. **Small n** for κ (10 items, sometimes 9)—treat as **pilot** stability, not definitive reliability.  
2. **Only ~17** human-filled rows per sheet vs **38** corpus rows—incomplete coverage.  
3. **Register κ vs LLM** is **not interpretable** while the LLM column is **constant**.  
4. **Pooled** human–LLM counts **repeat** the same items across annotators—good for “total judgments vs model,” not for independent items.

---

## 6. Regenerate numbers

```bash
cd "/path/to/Project"
python analytics/group_annotation_agreement.py
```

Outputs land in `metrics/` (see `metrics/GROUP_ANNOTATION_README.md`).
