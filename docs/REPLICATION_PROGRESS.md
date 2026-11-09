# Replication Progress

**Goal:** faithful methodological replication of Öğdü/Arslanoğlu/Karaköse, IEEE Access 2025 (DOI `10.1109/ACCESS.2025.3613340`), building on the completed prototype.
**Rule of thumb:** research fidelity, reproducibility, reliability and honest evaluation > high accuracy. Never claim the paper's 94/88/84 unless our results genuinely support it.

## Status board

| Phase | Description | Status | Notes |
|---|---|---|---|
| 1 | Paper analysis + repository audit → `REPLICATION_SPECIFICATION.md`, `REPLICATION_GAP_ANALYSIS.md` | ✅ **COMPLETE** | 26 gap entries; 24 paper artifacts extracted with page/section citations |
| 2 | Architecture & per-agent model configuration (shared Gemini default, per-agent overrides, fallback, per-agent model logging) + git checkpoint | ✅ **COMPLETE** | checkpoint `9e507fd`; `AGENT_CONFIG` + provider dispatch + Gemini fallback + `model_trace` (details below) |
| 3 | Clinical text processing (staged entities/associations + ICD-10-CM coding) | ⏭️ **NEXT** | SNOMED deferred (license decision); ICD-10-CM via free NLM API |
| 4 | Biomedical corpus & RAG (term boosting, Fig-5 fallback, metadata, optional HF symptom-disease collection) | ⏸️ queued | existing index must not be touched |
| 5 | External evidence scanning (Serper if key provided, optional NICE, failure isolation) | ⏸️ queued | needs your free Serper key decision |
| 6 | Clinical fusion & adaptive optimization (+ `web` mode) | ⏸️ queued | confidence weights stay ours, labeled |
| 7 | Experimental evaluation (micro metrics, ablation, iteration sweep, 3 datasets, MedBullets free-text) | ⏸️ queued | **API spend approval required before runs** |
| 8 | Web interface report upgrades (Fig. 6 fields) | ⏸️ queued | |
| 9 | Final replication analysis & documentation (README rewrite, deviation ledger) | ⏸️ queued | |

## What Phase 1 delivered (2026-11-09)

- `docs/REPLICATION_SPECIFICATION.md` — everything the paper specifies (five agents, model assignments, CrewAI, clinical text pipeline, SNOMED/ICD-10, corpus construction, BioBERT, chunking, Qdrant, fallback, external sources, evidence weighting, fusion, confidence, optimizer, iterations, web interface, datasets, baselines, RAG/web-only, iteration & ablation experiments, metrics, published results, hardware/timing, limitations) — each with §/page/journal-page citations, Tables 1–7 and Figures 1–6 OCR-verified.
- `docs/REPLICATION_GAP_ANALYSIS.md` — 26 requirements classified (FULLY / PARTIALLY / NOT / CANNOT BE REPRODUCED) against **source code**, with missing functionality, files to modify, external dependencies, feasibility, plus the phased plan.
- Key audit facts: 6 items FULLY implemented (agents, embeddings, retrieval, evidence hierarchy, optimizer mechanics, …), 9 PARTIAL, 6 NOT implemented, 4 CANNOT be reproduced from the paper (confidence formula, term-boost algorithm, paper corpus/test IDs, cross-case learning).

## Prototype baseline (do not overwrite)

| Configuration | Result | Source |
|---|---|---|
| Baseline | 17/20 (85%) | `results/eval_medqa_20261026_223318.csv` |
| RAG-only | 17/20 (85%) | same |
| Full pipeline | 18/20 (90%) | same |

New replication runs will use the `repl_` filename prefix so they can never be confused with these files.

## Decisions made (2026-11-09, all six answered before Phase 2)

1. **Plan approved** — phased implementation (Phases 2–9) with git checkpoint commit `9e507fd` before any code change. ✅ executed.
2. **CrewAI:** keep explicit Python orchestration — documented deviation (behavior matches Fig. 2; CrewAI adds heavy dependencies for zero behavioral gain).
3. **External sources (Phase 5):** Serper + NICE. Google Scholar skipped (no official API). User registers the free Serper key themselves → `SERPER_API_KEY=` in `.env` (never in chat).
4. **SNOMED CT:** deferred (license). **ICD-10-CM now** via the free NLM Clinical Tables API (Phase 3). Only codes actually returned may be claimed.
5. **Corpus (Phase 4):** add the public HF Symptom-Disease dataset (paper ref [30]) as a **new** Qdrant collection; skip the 1,000-PDF/OCR pipeline. Existing index untouched.
6. **Phase 7 API budget:** staged — **ablation first (~700 calls)**, results reviewed, then a separate go-ahead for the iteration sweep and the PubMedQA/MedBullets 50-question runs.

## Phase 2 delivered (2026-11-09)

- **`src/config.py`** — `AGENT_CONFIG` for six stages (`clarifier`, `rag_analyzer`, `evidence_scanner`, `fusion`, `optimizer`, `select`): per-agent `<AGENT>_PROVIDER` / `<AGENT>_MODEL` env overrides, shared `LLM_PROVIDER` default, shared Gemini default model. No keys in code — each provider reads `<PROVIDER>_API_KEY` from `.env`.
- **`src/llm.py`** — provider dispatch: native Gemini adapter (retry/backoff/model-chain preserved) + OpenAI-compatible adapter (openai/groq/openrouter/together/mistral/deepseek/xai/cerebras/perplexity, or any name + `<PROVIDER>_BASE_URL`); any provider failure → **graceful fallback to the Gemini chain** with the reason logged; thread-local per-call **model trace** (agent, requested vs executed provider/model, cache-hit, fallback flag+reason, duration, status). Cache key format unchanged → all existing cache entries still hit.
- **Agent tagging** — all 5 agent modules + `pipeline._select` pass `agent=...` to `generate_json`.
- **`src/pipeline.py`** — `run_case` resets/collects the trace and stores `log["model_trace"]` + `parameters.agent_config` in every execution log (all three modes).
- **`.env.example`** — commented per-agent override block (no real keys).
- **Tests (all PASS):** ① config/import smoke (`AGENT_CONFIG` = 6× gemini/flash-lite) ② broken-provider fallback (requested `localtest` → executed `gemini`, `fallback=True` + reason in trace) ③ cached full-pipeline `run_case` on `medqa_2` — correct answer C, per-agent `model_trace` recorded across 10 calls (clarifier/rag/fusion/optimizer/select) ④ scanner tag test (LLM query built, `agent=evidence_scanner`, 4 PubMed records) ⑤ saved-case demo replay `/api/cases/medqa_347` → 200 + full case JSON ⑥ `py_compile` clean on all touched modules.
- **Note:** `evidence_scanner` LLM calls appear in traces only when hypotheses exist and `SCANNER_LLM_QUERY=1` (default); for `medqa_2` the RAG stage found 0 evidence-backed hypotheses → scanner correctly had nothing to scan.
- **README updated:** §4 per-agent configuration + `model_trace`, §12 status, §13 deviation row, scanner default flag corrected.

## Log

- **2026-11-09** — Phase 1 complete: full paper re-extraction (all 7 tables + Figs 1–6 OCR), full source audit (15 source files read), spec + gap analysis written. No source code modified in Phase 1.
- **2026-11-09** — Phase 2 complete: git checkpoint `9e507fd`; per-agent model configuration (shared Gemini default, per-agent provider/model overrides in `.env`, automatic Gemini fallback, per-call `model_trace`); 6/6 tests pass; docs (README, PROGRESS, `.env.example`) updated. No results/index/`.env` touched.
