# Method notes — what we measured and how

Short reference for the report and for Q&A. Every number in the slides comes from one of
the commands or files listed here.

## Data

| Dataset | Source | Frozen sample | Scoring |
|---|---|---|---|
| MedQA | `bigbio/med_qa`, 5-option USMLE-style | 50 items, seed 42 | exact match on option letter |
| PubMedQA | `qiaojin/PubMedQA`, `pqa_labeled` | 50 items, seed 42 | paper's rule: cosine ≥ 0.80 |
| MedBullets | `LangAGI-Lab/medbullets` | 50 items, seed 42 | paper's rule: cosine ≥ 0.80 |

IDs are stable and live in `data/samples/*.jsonl`. The 20-question run uses the first 20
rows in file order; it is not re-sampled after seeing results.

The paper says 50 questions per dataset but never publishes IDs or the sampling seed, so
exact reproduction of its test sets is impossible.

## Configurations

| Mode | What runs | LLM calls per case (typical) |
|---|---|---|
| `baseline` | one prompt: question + options → option letter | 1 |
| `rag` | BioBERT/Qdrant top-5 → one selection prompt | 1 + retrieval |
| `full` | Clarifier → (RAG Analyzer → PubMed Scanner → Fusion) × iterations → selection | 5–7 per iteration |

`full` is the paper's configuration. It is not "RAG plus a bigger prompt": the Fusion
stage explicitly compares two evidence streams, and the Optimizer can rewrite the query
when confidence < τ.

## Scoring

- **MCQ:** predicted letter must equal the gold letter. Blank or malformed output = wrong.
- **Free text:** cosine between predicted and gold answer embeddings ≥ 0.80. This is the
  paper's own rule and is *lenient* (a short paraphrase can pass). It is labelled as such
  wherever it appears; it is not the standard PubMedQA/MedBullets protocol.
- Errors are recorded with the exception text, counted as wrong, never dropped.
- n = 20 per mode. A 1-question difference is 5 percentage points, so we report counts
  (e.g. 17/20) alongside percentages and avoid significance claims.

## Results (20 MedQA questions, run 2026-10-26 22:33)

| Configuration | Correct | Accuracy | Errors | Mean confidence | Mean iterations |
|---|---|---|---|---|---|
| Baseline | 17/20 | 85.0% | 0 | — | 0 |
| RAG only | 17/20 | 85.0% | 0 | — | 1 |
| Full | 18/20 | 90.0% | 0 | 0.791 | 1.5 |

- Full vs baseline: 17 both correct, 0 baseline-only, 1 full-only (`medqa_107`), 2 neither.
- RAG vs baseline: identical correctness; same three errors (`107`, `202`, `376`).
- Both full errors were high-confidence (`202` 0.855, `376` 0.849).
- The first RAG run had 3 Qdrant concurrency failures; the fixed run has 0 errors.
- Latency: use the cold-cache first run (baseline 28.8 s, RAG ~32 s, full 144.1 s).
  The clean rerun was cached (full 13 s/case) and is not deployment latency.

## Time-boxed settings used for the evaluation

| Setting | Value | Reason |
|---|---|---|
| `MAX_ITERS` | 2 (paper: 3) | 22–33 s per LLM call; 3 cycles × 20 cases exceeds the slot |
| `SCANNER_LLM_QUERY` | 0 | saves up to 3 calls/iteration; scanner uses condition name as query, with retry fallback |
| `TOP_K` | 5 | paper value |
| `TAU` | 0.65 | paper value |
| `LLM_TEMPERATURE` | 0 | determinism; every call cached by prompt hash |

These are deviations, not hidden knobs. They are printed in every audit log under
`parameters` and repeated on the deviations slide.

## Reproducibility

- `results/llm_cache/<sha256>.json` stores model, prompt and response for every call.
  Reruns hit the cache, so the demo is fast and the numbers do not drift.
  **Latency caveat:** cached reruns are much faster than the original run. The
  latency figures on the results slide come from the first, uncached evaluation
  (baseline 28.8 s/case, full 144.1 s/case); do not quote rerun timings as deployment
  latency.
- `results/execution_logs/<case>.json` stores the full trace for single cases.
- `results/eval_medqa_<stamp>.csv` stores one row per (question, mode), including failures.
- `scripts/summarize_eval.py` rebuilds the slide table and the paired comparison.

## Known measurement weaknesses

1. **Lexical retrieval check.** `scripts/eval_retrieval.py` looks for the gold answer
   string in retrieved text with exact token overlap ≥ 0.6. It has no stemming or synonym
   handling and scores 0/20 even when retrieved passages are clinically relevant. Treat it
   as a lower bound, not as retrieval accuracy.
2. **Confidence is a heuristic.** The paper does not publish its formula. Ours is linear
   in four logged components and is not calibrated; a confidence of 0.66 does not mean
   "66% chance correct".
3. **One model for five roles.** Role separation is prompt-level, not model-level, so
   errors can correlate across agents.
4. **Corpus scale.** 24,935 chunks against the paper's claimed 200k / 18M embeddings;
   retrieval recall is therefore not comparable.
5. **Free-text scoring is lenient.** PubMedQA/MedBullets numbers under θ = 0.80 should not
   be compared with literature that uses exact match or token F1.
6. **Live evidence is not deterministic.** PubMed records can change between runs, which
   changes the Fusion prompt. Accuracy was stable across our two runs (18/20), but one
   case's iteration count and another's confidence changed because the web stream changed.
   Only the LLM responses are cached; the web search is not.
