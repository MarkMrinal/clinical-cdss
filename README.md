# Clinical CDSS replication — adaptive multi-agent LLM + biomedical RAG

Student replication of:

> Öğdü, Ç. U., Arslanoğlu, K., Karaköse, M. *An Adaptive Multi-Agent LLM-Based Clinical
> Decision Support System Integrating Biomedical RAG and Web Intelligence.*
> IEEE Access, vol. 13, 2025. DOI: 10.1109/ACCESS.2025.3613340

Research/education prototype only. **Not a medical device. Do not use for patient care.**

---

## What is implemented

| Paper component | This repo | Fidelity |
|---|---|---|
| 5 agents (Clarifier, RAG Analyzer, Scanner, Fusion, Optimizer) | `src/clarifier.py`, `rag_analyzer.py`, `evidence_scanner.py`, `fusion.py`, `optimizer.py` | roles reproduced |
| BioBERT sentence-transformer, 768-d | `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb` | identical model id |
| Qdrant vector DB, cosine, top-k | `src/retrieval.py` (`QdrantClient(path=...)`, cosine) | identical engine, embedded mode |
| chunk 512 / overlap 64 | `scripts/build_index.py` | as reported |
| multi-level retrieval fallback + term emphasis | `search_with_fallback` | simplified |
| τ = 0.65, max 3 feedback iterations | `src/config.py`, `src/pipeline.py` | as reported |
| 5 different LLMs | one Gemini model for all roles | **deviation** (single API key) |
| Gemini-2.0-Flash (paper) | `gemini-flash-lite-latest` | **forced** — 2.0/2.5 retired, 3.8-flash 503 |
| Serper + Scholar + NICE + 1,000 PDFs | PubMed E-utilities + MedRAG textbooks (18 texts) | **deviation** (free, citable) |
| CrewAI orchestration | explicit Python pipeline | **deviation** (transparent, testable) |
| SNOMED CT / ICD-10 coding | LLM extraction only | **not implemented** |
| Confidence formula | documented heuristic | **not published in paper** |

### Confidence heuristic (ours, not the paper's)

```
confidence = 0.35*retrieval_similarity
           + 0.25*evidence_breadth
           + 0.20*web_support
           + 0.20*cross_source_agreement
```

All four components are logged per iteration in `results/execution_logs/*.json`.
The paper reports the 0.65 threshold but never the formula, so this is explicitly a
replication approximation — **not** a calibrated probability.

---

## Setup

```powershell
cd "E:\D Drive\clinical-cdss"
pip install -r requirements.txt
```

Put your Google AI Studio key in `.env`:

```
GEMINI_API_KEY=...
```

## Build the knowledge base (no API key needed)

```powershell
python scripts\fetch_pubmed.py          # ~2,800 abstracts, free NCBI API
python scripts\build_index.py           # chunk -> BioBERT -> Qdrant (biomedical)
python scripts\build_textbook_index.py  # MedRAG textbooks -> Qdrant (textbooks)
python scripts\smoke_retrieval.py       # inspect cited passages
```

## Freeze the benchmarks (no API key needed)

```powershell
python scripts\sample_datasets.py
```

Writes `data/samples/{medqa,pubmedqa,medbullets}_sample.jsonl` with stable IDs.
The paper evaluates 50 questions per dataset but never says which; we use seed 42.

## Run

```powershell
python main.py --case medqa_12                    # full pipeline on one MCQ
python main.py --question "45M, fever, cough..."  # free-text case
python main.py --question "..." --mode rag        # retrieval only
```

## Local web demonstration

Start the local replay viewer with:

```powershell
python web_demo.py
```

Then open `http://127.0.0.1:8765`. The default view accepts a free-text case and submits it
to the existing full pipeline; the result shows the assessment, rationale, confidence,
iteration trace, retrieved passages, and PubMed references. Live analyses use the configured
Gemini API and PubMed and write a local audit log. Use fictional, non-identifiable input.

The **Saved demonstrations** tab replays `medqa_347` (one iteration) and `medqa_107`
(adaptive re-query) from `results/execution_logs/` without making API calls. Their original
multiple-choice options are collapsed under the benchmark prompt, since those are benchmark
records rather than inputs a user must choose from. Keep the server running during the demo
and stop it with Ctrl+C afterward.

## Evaluate

```powershell
python evaluation\run_eval.py --dataset medqa --n 20 --modes baseline,rag,full
```

Outputs `results/eval_medqa_<timestamp>.csv` + `.json` with per-case rows, accuracy,
mean confidence, iteration count and latency. Failures are recorded, never imputed.
`python scripts\summarize_eval.py` prints a slide-ready table and a paired
baseline-vs-full comparison from the newest CSV.

## Results (20 MedQA questions, seed 42)

| Configuration | Correct | Accuracy | Errors | Mean confidence | Mean iterations |
|---|---|---|---|---|---|
| Baseline (LLM only) | 17/20 | 85.0% | 0 | — | 0 |
| RAG only | 17/20 | 85.0% | 0 | — | 1 |
| Full (RAG + PubMed + fusion + loop) | **18/20** | **90.0%** | 0 | 0.791 | 1.5 |

The full pipeline fixed `medqa_107` (wrong at baseline and RAG-only, correct after two
iterations) and broke no baseline-correct case. RAG alone changed no decision. With n=20,
a one-question difference is not statistically significant.

Latency from the first uncached run: 28.8 s/case (baseline), ~32 s/case (RAG),
144.1 s/case (full). Cached reruns are much faster and must not be quoted as deployment
latency.

Clean run: `results/eval_medqa_20261026_223318.csv` / `.json`. The first RAG run had three
Qdrant concurrency failures; the client-initialisation race was fixed and the rerun has
zero errors. Live PubMed results can change between runs, so full-mode iteration counts
and confidence are not bit-for-bit reproducible even with the LLM cache.

---

## Reproducibility

- Every LLM call is cached in `results/llm_cache/` (keyed by model, prompt, temperature and
  JSON mode). Reruns of identical prompts cost nothing and return the same text.
  Live PubMed searches are not cached, so the full pipeline is not fully deterministic
  across days even when the LLM responses are.
- Every case writes an audit log to `results/execution_logs/`: clarifier output, retrieved
  passages with PMIDs/URLs, PubMed records, hypotheses, web queries, per-iteration
  confidence components, stop reason, optimizer feedback and final selection.
- Question IDs are frozen in `data/samples/` and must not be resampled between runs.

## Known deviations and limitations

1. **No model heterogeneity.** One Gemini model plays all five roles (single API key).
1b. **Model forced to change.** The paper's `gemini-2.0-flash` returns "no longer available"
   on current keys; `gemini-2.5-flash` is closed to new users; `gemini-3.8-flash` returns
   503 "high demand" or times out. The only model that answered reliably on our key was
   `gemini-flash-lite-latest` (~22–33 s/call), which we log in every audit trail.
2. **Corpus is abstracts + textbooks, not 1,000 PDFs.** Two Qdrant collections: ~2.8k
   PubMed abstracts (11,244 chunks at 512/64) and a balanced 780-passage sample from each
   of 18 MedRAG textbooks (13,691 passages, native segmentation). Textbooks were added
   because a PubMed-only index is thin for exam-style questions and the paper's corpus
   explicitly included textbooks. A crude lexical check finds the gold answer string in
   the top-5 for **0/20** MedQA questions both before and after adding textbooks; that
   check has no stemming and misses paraphrases, so it understates retrieval quality —
   it is reported as a limitation, not as a success metric.
   The paper's "18M embeddings from 200k chunks" is arithmetically inconsistent and was
   not treated as a target.
3. **Web evidence = PubMed only.** No Google Scholar scraping, no NICE guideline API.
4. **No SNOMED CT / ICD-10 mapping** (licence required).
5. **No CrewAI.** The feedback loop is orchestrated explicitly in `src/pipeline.py`.
6. **Confidence is a heuristic**; the paper's formula is not published.
7. **Evaluation sample is small** and, for PubMedQA/MedBullets, the paper's own
   free-text cosine-similarity scoring (theta = 0.80) differs from the standard
   benchmark protocol. Report both where possible.
8. **MedQA questions are exam items**, not patient narratives; the Clarifier therefore
   sees a question stem rather than free text. The paper has the same property.

## Citation

```bibtex
@article{ogdu2025adaptive,
  title   = {An Adaptive Multi-Agent LLM-Based Clinical Decision Support System
             Integrating Biomedical RAG and Web Intelligence},
  author  = {Öğdü, Çağatay Umut and Arslanoğlu, Kübra and Karaköse, Mehmet},
  journal = {IEEE Access},
  volume  = {13},
  year    = {2025},
  doi     = {10.1109/ACCESS.2025.3613340}
}
```
