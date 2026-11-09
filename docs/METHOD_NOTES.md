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
| `web` | Clarifier → knowledge-based hypothesis proposal → Evidence Scanner (PubMed + NICE + Serper) → selection; **no local index, no fusion loop** | 3–4 + web fetches |
| `full` | Clarifier → (RAG Analyzer → Evidence Scanner → Fusion) × iterations → selection | 5–7 per iteration |

`full` is the paper's configuration. It is not "RAG plus a bigger prompt": the Fusion
stage explicitly compares two evidence streams, and the Optimizer can rewrite the query
when confidence < τ.

The two ablation arms (`rag`, `web`) are single-pass by design: Fusion exists to
cross-check **two** streams, so with one stream there is nothing to fuse — symmetric
with each other and with the paper's own ablation structure. `confidence` is `None`
in both, so a mean confidence is only ever reported for `full`.

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

- **Retrieval backend:** the reported evaluation ran on the **local** Qdrant index
  (`data/raw/qdrant_index`, then 24,935 vectors; 34,206 after replication Phase 4). A later deployment-preparation change put a
  Qdrant Cloud URL in `.env`; that cluster became unreachable, so on 2026-11-09 the two
  `QDRANT_*` lines were blanked (with approval) to restore local mode — the same backend
  the published numbers came from. The evaluation files predate the cloud change, so no
  result was affected.
- `results/llm_cache/<sha256>.json` stores model, prompt and response for every call.
  Reruns hit the cache, so the demo is fast and the numbers do not drift.
  **Latency caveat:** cached reruns are much faster than the original run. The
  latency figures on the results slide come from the first, uncached evaluation
  (baseline 28.8 s/case, full 144.1 s/case); do not quote rerun timings as deployment
  latency.
- `results/execution_logs/<case>.json` stores the full trace for single cases. Since
  replication Phase 2 (2026-11-09) it also records `model_trace` — one entry per LLM
  call with the agent, requested vs. actually executed provider/model, cache-hit flag,
  fallback flag + reason, and duration — plus `parameters.agent_config`. The two saved
  demo logs (`medqa_107`, `medqa_347`) predate Phase 2 and therefore have no
  `model_trace`.
- `results/repl_eval_<dataset>_<stamp>.csv` stores one row per (question, mode),
  including failures, plus evidence columns (`n_passages`, `n_web_records`,
  `evidence_sources`). Every replication run uses the **`repl_` prefix**; the frozen
  prototype benchmark keeps its original `eval_medqa_<stamp>` names. Each run also
  writes `..._evidence.jsonl` — one line per (question, mode) with the passages, web
  records (title/URL/source/level), Fig. 5 rungs, stop reasons, optimizer feedback,
  confidence factors, per-diagnosis rows and the model trace (`--evidence 0` skips it).
- `scripts/summarize_eval.py` rebuilds the slide table and the paired comparison
  (reads both `eval_*` and `repl_eval_*` prefixes).

## Known measurement weaknesses

1. **Lexical retrieval check.** `scripts/eval_retrieval.py` looks for the gold answer
   string in retrieved text with exact token overlap ≥ 0.6. It has no stemming or synonym
   handling and scores 0/20 even when retrieved passages are clinically relevant. Treat it
   as a lower bound, not as retrieval accuracy. It is used only to *report* retrieval
   behaviour (e.g. after Phase 4: mean top-1 0.644, and which Fig. 5 rung returned the
   passages — 8/20 @0.65, 11/20 @0.55, 1/20 @0.45); no threshold was ever chosen from it.
2. **Confidence is a heuristic.** The paper does not publish its formula. Ours is linear
   in four logged components and is not calibrated; a confidence of 0.66 does not mean
   "66% chance correct".
3. **One model for five roles.** Role separation is prompt-level, not model-level, so
   errors can correlate across agents. Since replication Phase 2 the architecture
   supports per-agent provider/model overrides (`.env`) with automatic Gemini fallback
   and per-call `model_trace` logging, but all five roles still run the same Gemini
   model by default (single available key) — so the correlation weakness remains.
4. **Corpus scale.** 34,206 chunks (Phase 4) against the paper's claimed 200k / 18M
   embeddings; retrieval recall is therefore not comparable. The frozen numbers in §Results
   were measured on the earlier 24,935-chunk two-collection index, before τ filtering,
   term boosting and the third collection existed — they are not re-measured here.
5. **Free-text scoring is lenient.** PubMedQA/MedBullets numbers under θ = 0.80 should not
   be compared with literature that uses exact match or token F1.
6. **Live evidence is not deterministic.** PubMed and NICE records can change between
   runs, which changes the Fusion prompt. Accuracy was stable across our two runs
   (18/20), but one case's iteration count and another's confidence changed because the
   web stream changed. LLM responses are cached; **Serper responses are cached too**
   since Phase 5 (`results/serper_cache/`), but PubMed and NICE are always live, so a
   full run is still not bit-reproducible. The historical numbers in §Results predate
   Phase 5 entirely (their scanner returned PubMed records only).
8. **Web pages are ranked conservatively.** The paper weights guidelines above case
   reports but publishes no weights for ordinary web pages, so we place NICE guidance at
   level 1, PubMed/PMC links at level 3 and every other web page at level 4 (below case
   reports, because the study design is unverifiable). That is our choice, not the
   paper's, and it affects only record ordering/display — the confidence heuristic does
   not read the level.
7. **Term boosting is an approximation (Phase 4).** Table 2 says boosting is "Active" but
   the paper never publishes the mechanism. `_boost_query()` (append absent priority
   terms, repeat present ones once, skip negations, ≤5 terms / 400 chars) is our reading,
   not the paper's algorithm — so any accuracy difference attributable to boosting cannot
   be attributed to *the paper's* boosting. The τ rungs 0.65/0.55/0.45 come from
   Table 2 / Fig. 5 annotations and were never tuned against evaluation questions; the
   "no-threshold safety return" at the end of the ladder is also ours (Fig. 5 does not
   define what happens when all three strategies fail).
9. **Web-only hypothesis proposal is ours (Phase 6).** The paper reports a web-only
   configuration but never says how per-hypothesis search terms are formed once the RAG
   Analyzer is gone. Our `web` mode therefore makes one knowledge-based LLM call
   (`rag_analyzer.propose_without_retrieval`) that proposes up to 3 candidate conditions
   from the clinical input alone, falling back to the Clarifier's key concepts if the
   model fails. Any web-only accuracy number is therefore *our* arm, not the paper's.
10. **Per-diagnosis confidence rows are a deterministic rule (Phase 6).** Each report row
    reuses the aggregate weights with the web terms evaluated for that one diagnosis
    (base + 0.20·web + 0.20·concordance). The rows average to the aggregate confidence
    only when condition names are unique (duplicate conditions would each get the same
    web flag, so the mean can shift). The rule is ours; the paper asks for a confidence
    per diagnosis but publishes no rule for how it is computed.
