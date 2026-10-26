# Two demos, using the real saved logs

Both commands replay against the LLM disk cache, so they finish in seconds. The
retrieval and PubMed calls are still real. If you want the original cold-cache timing,
the single MedQA case took 100.6 s.

## Demo A — normal successful workflow

```powershell
cd "E:\D Drive\clinical-cdss"
python main.py --case medqa_347
python scripts\show_case.py medqa_347
```

What to point at:

- **Question:** 5-option MedQA item (primary hyperparathyroidism).
- **Clarifier:** symptoms, test results, key concepts, missing information.
- **Iteration 1:** `hypotheses: ['Primary hyperparathyroidism']`, 5 RAG passages,
  4 PubMed records; `stop: confidence_threshold_met`.
- **Confidence:** 0.8577, above τ = 0.65, so no re-query.
- **Final:** A, primary hyperparathyroidism, one-sentence rationale.
- **Log:** `results/execution_logs/medqa_347.json` contains the passages with PMIDs and
  the PubMed records, not just counts.

The first RAG passage is an endocrinology review (PMID 40690933); the first PubMed record
is "Primary hyperparathyroidism" (PMID 32797471, evidence level 2).

## Demo B — adaptive correction

```powershell
python main.py --case medqa_107
python scripts\show_case.py medqa_107
```

What to point at, in order:

1. **First pass is weak.** `confidence=0.4499`, `hypotheses: []`, `web_records: 0`.
   The retrieved passages are mostly irrelevant (primary aldosteronism, thyroid, androgen
   insensitivity). This is an honest failure, not a staged example.
2. **Optimizer explains why.** Its reason: the query was a "keyword dump" that missed the
   underlying diagnosis, Klinefelter syndrome (47,XXY), and paternal meiotic non-disjunction.
   Refined query: `Klinefelter syndrome paternal meiotic non-disjunction meiosis I spermatogenesis error`.
3. **Second pass works.** `hypotheses: ['Klinefelter syndrome']`, 4 PubMed records,
   `confidence=0.8277`, `stop: confidence_threshold_met`.
4. **Final:** A, primary spermatocyte — the stage of paternal meiosis I non-disjunction.

The important nuance: the first pass did **not** produce a wrong final answer; it produced
no usable hypothesis or web evidence. The optimizer fixed the query, not the answer text.
Say it that way.

## If asked "was that real or cached?"

The answers and retrieved evidence are real and saved in the logs. The LLM responses are
replayed from the disk cache, which is why the commands take ~17 s instead of ~100 s.
The cache stores the exact prompt and model for every call in `results/llm_cache/`.

## Optional third demo — synthetic pneumonia

```powershell
python scripts\demo_case.py
```

This is a synthetic case, not a benchmark item. It shows the loop correcting a wrong first
hypothesis (pseudomesotheliomatous carcinoma → pneumonia) but confidence stays at 0.4794,
below τ, so it stops at the 2-iteration cap. Use it only if you want a second correction
story; `medqa_107` is stronger because it is a real benchmark item.
