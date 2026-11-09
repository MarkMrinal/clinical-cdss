# Replication Progress

**Goal:** faithful methodological replication of Öğdü/Arslanoğlu/Karaköse, IEEE Access 2025 (DOI `10.1109/ACCESS.2025.3613340`), building on the completed prototype.
**Rule of thumb:** research fidelity, reproducibility, reliability and honest evaluation > high accuracy. Never claim the paper's 94/88/84 unless our results genuinely support it.

## Status board

| Phase | Description | Status | Notes |
|---|---|---|---|
| 1 | Paper analysis + repository audit → `REPLICATION_SPECIFICATION.md`, `REPLICATION_GAP_ANALYSIS.md` | ✅ **COMPLETE** | 26 gap entries; 24 paper artifacts extracted with page/section citations |
| 2 | Architecture & per-agent model configuration (shared Gemini default, per-agent overrides, fallback, per-agent model logging) + git checkpoint | ⏸️ **AWAITING APPROVAL** | see Decisions below |
| 3 | Clinical text processing (staged entities/associations + ICD-10-CM coding) | ⏸️ queued | SNOMED pending license decision |
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

## Decisions requested (block Phase 2)

1. Approve plan as-is? (incl. git checkpoint commit before code changes)
2. CrewAI: keep Python orchestration (recommended) or integrate real CrewAI?
3. Serper free key: will you register one? NICE pages: include or skip? (Scholar: skip — no official API.)
4. SNOMED CT: defer (license), implement ICD-10-CM now — OK?
5. Corpus: add public HF Symptom-Disease dataset as a new collection; skip 1,000-PDF/OCR pipeline — OK?
6. Phase 7 API budget (~1,500–2,500 flash-lite calls for sweep + ablation + 3 datasets): approve, reduce, or stage it?

## Log

- **2026-11-09** — Phase 1 complete: full paper re-extraction (all 7 tables + Figs 1–6 OCR), full source audit (15 source files read), spec + gap analysis written. No source code modified in Phase 1.
