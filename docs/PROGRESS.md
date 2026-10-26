# Progress report — replication status

**Base paper:** Öğdü, Arslanoğlu, Karaköse. *An Adaptive Multi-Agent LLM-Based Clinical
Decision Support System Integrating Biomedical RAG and Web Intelligence.* IEEE Access 13 (2025).
DOI 10.1109/ACCESS.2025.3613340

**Team:** 2 students · **Milestone:** working prototype (Tue 27 Oct 2026)

---

## Status

| Component | Status | Evidence |
|---|---|---|
| Paper analysis + architecture reconstruction | done | `docs/RELATED_WORK.md`, slides 2–3 |
| Repo scaffold, config, logging, LLM cache | done | `src/`, `results/llm_cache/` |
| PubMed corpus (30 clinical topics, 2 failed) | done | 2,784 abstracts → 11,244 chunks |
| Textbook corpus (MedRAG, 18 books) | done | 13,691 passages from 780/book sample |
| BioBERT embeddings + Qdrant index | done | 24,935 chunks, 768-d, cosine |
| Retrieval: dual-collection RRF + 3-level fallback | done | `textbook share of top-k = 40%` |
| Agent 1 Clinical Text Clarifier | done | `src/clarifier.py` |
| Agent 2 RAG Analyzer | done | `src/rag_analyzer.py` |
| Agent 3 Evidence Scanner (PubMed) | done | 4 records/query after query fix |
| Agent 4 Fusion + confidence | done | `src/fusion.py`, per-iteration components logged |
| Agent 5 Optimizer (τ=0.65, 2 iters in eval) | done | `src/optimizer.py` |
| Pipeline + audit logs | done | 2 validation runs, logs on disk |
| Benchmark samples (50/dataset, fixed IDs) | done | `data/samples/*.jsonl`, seed 42 |
| Evaluation harness (baseline / RAG / full) | done | `evaluation/run_eval.py` |
| 20-question MedQA evaluation | done | baseline 85%, RAG 85%, full 90%; 0 errors |
| Qdrant client concurrency fix + stress test | done | 8 threads / 16 calls, 0 failures, counts unchanged |

## Measured results

| Measurement | Value | Note |
|---|---|---|
| Index size | 24,935 chunks | 11,244 PubMed + 13,691 textbook |
| Retrieval latency | ~1–2 s/query | CPU BioBERT; model load ~30 s once |
| LLM latency | 22–33 s/call | `gemini-flash-lite-latest`, only working model |
| MedQA single case (`medqa_347`) | **correct**, conf 0.8577 | 1 iteration, 100.6 s |
| Adaptive MedQA case (`medqa_107`) | **correct after 1 requery** | pass 1: 0 hypotheses, 0 web records, conf 0.4499; pass 2: Klinefelter, 4 records, conf 0.8277 |
| Synthetic pneumonia case | **correct after 1 requery** | loop fixed a bad first hypothesis; 2 iters, 268 s, confidence 0.4794 (below τ) |
| MedQA 20 q, baseline | **17/20 (85.0%)** | 0 errors |
| MedQA 20 q, RAG only | **17/20 (85.0%)** | 0 errors; same three errors as baseline |
| MedQA 20 q, full pipeline | **18/20 (90.0%)** | 0 errors; mean conf 0.791, 1.5 iters |
| Paired full vs baseline | **+1 fixed, 0 regressions** | `medqa_107` (B → A after 2 iterations) |
| PubMed scanner | 4 records/condition | after condition-only query fix |
| Lexical gold-answer-in-top-k | 0/20 | crude word-overlap check; **understates** retrieval (no stemming, answer often paraphrased) |
| Textbook share of top-5 | 40% | proves RRF fusion is working |

The 0/20 lexical number is reported because it is the only automated check we have; it is
not evidence that retrieval is useless, and it did not improve when textbooks were added.

Latency figures are from the first, uncached run. The clean rerun used cached LLM responses
(13 s/case for full), so rerun timings are not deployment latency. Live PubMed records can
change between runs; accuracy was stable at 18/20 across both runs, but one case's iteration
count changed.

## Demo path (Tuesday)

```powershell
python scripts\smoke_retrieval.py                  # cited passages for one MedQA item
python main.py --case medqa_347                    # Demo A: normal, 1 iteration
python scripts\show_case.py medqa_347              # readable passages + PMIDs
python main.py --case medqa_107                    # Demo B: adaptive correction, 2 iterations
python scripts\show_case.py medqa_107              # show the failed first pass + refined query
python evaluation\run_eval.py --dataset medqa --n 20
```

Full demo script: `docs/DEMO.md`.

The audit log for any case is written to `results/execution_logs/<case>.json` and contains:
structured input, every retrieved passage with PMID/URL, every web query, per-iteration
confidence components, the optimizer's stated reason for requery, and the stop reason.

## Deviations from the paper (to put on a slide)

1. One Gemini model plays all five roles (paper: DeepSeek-R1, Gemini-2.0-Flash, Qwen2-72B, GPT-4o-mini).
2. Model id forced to `gemini-flash-lite-latest`: `gemini-2.0-flash` and `gemini-2.5-flash`
   are retired on current keys; `gemini-3.8-flash` returns 503 / times out.
3. Corpus = 2,784 PubMed abstracts + 13,691 textbook passages, not 1,000 PDFs / 200k chunks / 18M embeddings.
4. External evidence = PubMed E-utilities only (paper: Serper → Scholar, NICE).
5. No SNOMED CT / ICD-10 coding (licence); LLM extraction only.
6. Explicit Python orchestration instead of CrewAI.
7. Confidence formula is ours; the paper publishes only the 0.65 threshold.
8. Evaluation runs with `MAX_ITERS=2` and `SCANNER_LLM_QUERY=0` to fit the API time budget.

## Open questions for the final report

- The paper's "RAG-only 65.0% / web-only 70.0%" numbers equal two ablation rows in Table 7
  (remove RAG agent; remove Optimizer). The labels appear inconsistent — we measure all
  four configurations separately.
- Table 6 reports 86.3% → 94.7% over iterations, which is impossible on 20 exact-match
  questions; the protocol is undocumented.
- Free-text scoring in the paper (cosine ≥ 0.80) is lenient versus standard benchmark protocols.
