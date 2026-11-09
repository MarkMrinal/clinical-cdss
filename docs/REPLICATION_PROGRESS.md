# Replication Progress

**Goal:** faithful methodological replication of Öğdü/Arslanoğlu/Karaköse, IEEE Access 2025 (DOI `10.1109/ACCESS.2025.3613340`), building on the completed prototype.
**Rule of thumb:** research fidelity, reproducibility, reliability and honest evaluation > high accuracy. Never claim the paper's 94/88/84 unless our results genuinely support it.

## Status board

| Phase | Description | Status | Notes |
|---|---|---|---|
| 1 | Paper analysis + repository audit → `REPLICATION_SPECIFICATION.md`, `REPLICATION_GAP_ANALYSIS.md` | ✅ **COMPLETE** | 26 gap entries; 24 paper artifacts extracted with page/section citations |
| 2 | Architecture & per-agent model configuration (shared Gemini default, per-agent overrides, fallback, per-agent model logging) + git checkpoint | ✅ **COMPLETE** | checkpoint `9e507fd`; `AGENT_CONFIG` + provider dispatch + Gemini fallback + `model_trace` (details below) |
| 3 | Clinical text processing (staged entities/associations + ICD-10-CM coding) | ✅ **COMPLETE** | staged Clarifier + `src/coding.py` (NLM ICD-10-CM); SNOMED deferred (details below) |
| 4 | Biomedical corpus & RAG (term boosting, Fig-5 fallback, metadata, optional HF symptom-disease collection) | ✅ **COMPLETE** | third collection `symptom_disease` (9,271 pts) + τ=0.65 + boosting + Fig. 5 ladder; 22/22 tests (details below) |
| 5 | External evidence scanning (Serper if key provided, optional NICE, failure isolation) | ✅ **COMPLETE** | Serper (live, key in `.env`) + **keyless NICE** + PubMed, per-source isolation, web summaries (details below) |
| 6 | Clinical fusion & adaptive optimization (+ `web` mode) | ✅ **COMPLETE** | three named §II-A-4 factors (weights ours, arithmetic unchanged) + per-diagnosis report + `web`-only arm; 59/59 tests + 20-ID regression 18/20 (details below) |
| 7 | Experimental evaluation (micro metrics, ablation, iteration sweep, 3 datasets, MedBullets free-text) | ⏭️ **NEXT** | **API spend approval required before runs** (Serper cache already warm for the MedQA 20) |
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
- **`src/pipeline.py`** — `run_case` resets/collects the trace and stores `log["model_trace"]` + `parameters.agent_config` in every execution log (all modes).
- **`.env.example`** — commented per-agent override block (no real keys).
- **Tests (all PASS):** ① config/import smoke (`AGENT_CONFIG` = 6× gemini/flash-lite) ② broken-provider fallback (requested `localtest` → executed `gemini`, `fallback=True` + reason in trace) ③ cached full-pipeline `run_case` on `medqa_2` — correct answer C, per-agent `model_trace` recorded across 10 calls (clarifier/rag/fusion/optimizer/select) ④ scanner tag test (LLM query built, `agent=evidence_scanner`, 4 PubMed records) ⑤ saved-case demo replay `/api/cases/medqa_347` → 200 + full case JSON ⑥ `py_compile` clean on all touched modules.
- **Note:** `evidence_scanner` LLM calls appear in traces only when hypotheses exist and `SCANNER_LLM_QUERY=1` (default); for `medqa_2` the RAG stage found 0 evidence-backed hypotheses → scanner correctly had nothing to scan.
- **README updated:** §4 per-agent configuration + `model_trace`, §12 status, §13 deviation row, scanner default flag corrected.

## Phase 3 delivered (2026-11-09)

- **`src/clarifier.py` — staged Clarifier (paper §II-A-1):**
  1. **Normalisation (deterministic):** `normalize_text()` — NFKC + whitespace cleanup + ~70 high-precision abbreviation expansions (HTN, SOB, DM2, PMHx, w/, y/o in both orders, …) with word-boundary safety (never rewrites inside longer tokens). The model receives the normalised text; it is stored as `normalized_text`.
  2. **NER + association extraction (zero-shot LLM):** new schema fields `entities[] {text, type}` using the paper's four classes (an honest `Other` bucket is kept if the model insists on another class) and `associations[] {entity_a, relation, entity_b}` (the paper's "numbness ↔ left arm" pattern); outputs sanitised in code.
  3. **Terminology mapping (deterministic):** new `src/coding.py`.
  4. **Structured output:** the original JSON contract preserved byte-for-byte — new fields are additive; downstream agents unchanged.
- **`src/coding.py` — ICD-10-CM via the free NLM Clinical Tables API** (no key, no cost): only codes NLM actually returns are reported (`system: ICD-10-CM` + `matched_on`); local re-rank + progressive token fallback ("community acquired pneumonia" → **J18.9, matched_on "pneumonia"**); disk cache `data/raw/icd10cm_cache.json` (gitignored); 0.2 s throttle; per-case budget `ICD10_MAX_QUERIES=8`; `ICD10_ENABLED=0` disables cleanly; every failure path degrades to `unmapped_concepts` + `coding_status` (ok/degraded/disabled) — never crashes. Medication/Test-Result entities are honestly unmapped (ICD-10-CM codes neither; SNOMED deferred by decision).
- **Config/env:** `ICD10_ENABLED`, `ICD10_MAX_QUERIES` in `src/config.py`; commented block in `.env.example`. **No new dependencies** (`requests` already present).
- **Known effect:** the Clarifier prompt/schema changed → full-mode LLM cache entries from earlier phases no longer match (expected; old cache files remain, results untouched).
- **Tests (all PASS):** ① normalisation units (incl. no partial-token matches) ② entity/association cleaning (4 classes + Other, bad shapes dropped) ③ live NLM coding: exact R06.02, fallback J18.9 with `matched_on`, medication/test reasons, no-match, query-budget ④ staged `clarify()` on a synthetic narrative (9 entities across all 4 classes, 6 associations, 5 codes, status ok) ⑤ `ICD10_ENABLED=0` path ⑥ full `run_case` free-text downstream (τ met at iter 1, conf 0.6673, codes retained, model_trace complete; a transient NCBI 502 degraded with warnings, no crash) ⑦ demo replay `/api/cases/medqa_347` 200 ⑧ `py_compile` clean.

## Phase 4 delivered (2026-11-09)

- **Third corpus source (paper ref [30]):** new `scripts/build_symptom_index.py` downloads the public HF dataset `dux-tecblic/symptom-disease-dataset` (7,043 patient-style narratives, 1,082 disease classes, cached under `data/raw/symptom_disease/`), builds passage `"<narrative> Condition: <disease>"`, splits 512/64 (paper Table 2), embeds with the same BioBERT, and upserts into a **new** collection `symptom_disease` — **9,271 points**. Safety: it refuses to overwrite a non-empty collection unless `--recreate`, and never touches `biomedical`/`textbooks` (which `build_index.py` would delete on rerun). `config.COLLECTIONS` now = `["biomedical", "textbooks", "symptom_disease"]` (11,244 + 13,691 + 9,271 = **34,206 chunks**).
- **Evidence threshold τ = 0.65 (Table 2):** `RETRIEVAL_THRESHOLD` config; `_raw_search` over-fetches ×5 and filters client-side (works in embedded and cloud Qdrant, version-proof). Hits below τ are inadmissible evidence.
- **Term boosting (Table 2: "Active"):** `_boost_query()` — appends absent priority terms, repeats present ones once, **never boosts negated terms** ("no …", "denies", …), max 5 terms / 400 chars; `TERM_BOOST=1` flag (needed for the Table-7 ablation arm). Priority terms come from the Clarifier (`key_concepts + symptoms`). **Labeled approximation — the paper never publishes the algorithm.**
- **Fig. 5 fallback ladder:** `search_with_fallback` now runs boosted search @0.65 → Strategy 1 threshold reduction @0.55 → @0.45 → Strategy 2 simplify query @0.45 → Strategy 3 remove symptoms @0.45 → **no-threshold safety return** (ours: downstream never gets zero passages). The old widen-k level was removed (not in Fig. 5). Each rung logs `{level, strategy, threshold, n_hits, best_score, query_used, ms}` + total `retrieval_ms`; full-mode runs store the trace per iteration (`iterations[i].retrieval_trace`); RAG mode keeps `retrieval.fallback_trace`. Hits now carry `length`.
- **RAG analyzer** passes `boost_terms` (concepts + symptoms) and `symptom_terms` (for Strategy 3) into retrieval.
- **Config/env:** `RETRIEVAL_THRESHOLD`, `TERM_BOOST` in `src/config.py` + commented `.env.example` block; `data/raw/symptom_disease/` gitignored. **No new dependencies.**
- **Measurement (not tuning — `scripts/eval_retrieval.py`, 20 MedQA items, lexical gold-answer check):** before = gold-in-top-k 0/20, mean top-1 0.638, textbook share 40%; after = 0/20, mean top-1 0.644, textbook share 31% (symptom collection now takes share). Ladder distribution: boosted_search 8/20, threshold_reduction@0.55 11/20, @0.45 1/20 — i.e. τ=0.65 genuinely filters and the fallbacks genuinely fire. This lexical check has no stemming and understates retrieval (README §13 limitation 5); it was **not** used to choose any threshold.
- **Tests (22/22 PASS):** config defaults; boosting rules (absent/present/negation/5-term/400-char caps); symptom stripping; normal query (level-1 strategy+τ, hits respect the returning level's threshold, `length` present, `retrieval_ms`); weak query fires the full ladder incl. safety return; all levels record threshold/ms/best_score; `symptom_disease` count > 0; full `run_case` (confidence, `model_trace`, per-iteration `retrieval_trace` + `retrieval_ms`); demo log intact; RAG-mode replay of `medqa_347` (5 passages ≥0.65 via `boosted_search`); `py_compile` clean.

## Phase 5 delivered (2026-11-09)

- **`src/serper.py` — Serper.dev client (paper §II-A-3 "a web search API like Serper"):** POST `google.serper.dev/search` with `X-API-KEY` read from `.env` via config (**never logged, never printed, never written to any file**); 401/402/429 mapped to named reasons; **disk cache** `results/serper_cache/` (gitignored) so identical queries cost 0 credits and reruns return identical pages (reproducibility); 0.4 s throttle; contract `search() -> (records, error_reason)` — **never raises, never fabricates**.
- **`src/nice.py` — NICE guideline search, KEYLESS (free, no quota):** queries `nice.org.uk/search` directly instead of burning Serper credits; defensive parse of the `SearchCardList_listItem` / `card__heading` / `card__summary` / `headinglink` markup with every field optional → a NICE redesign degrades to `nice_no_results_or_layout_changed`, never a crash; NICE result types mapped onto the paper's hierarchy (guideline/quality standard/technology appraisal → level 1, evidence summary/advice → 2, other → 3); 0.6 s throttle.
- **`src/evidence_scanner.py` — per-hypothesis fan-out to 3 sources:** PubMed (existing) + NICE + Serper, each in its own try/except; records tagged `source ∈ {pubmed, nice, web}` and carry `evidence_level / evidence_label / weight / condition / query_used`; **web weighting is a documented conservative choice** (PubMed/PMC link → 3 `web_literature`, other web page → 4 `web_page`, below case reports since design is unverifiable); return adds `sources` (per-source status map) and keeps `search_failures` with per-source reasons.
- **Web summaries (paper: *"derives text summaries pertinent to each hypothesis"*):** one LLM pass (`SCANNER_WEB_SUMMARY=1`, default on) turns the NICE/web pages into a 2–3 sentence summary per condition citing `[W#]` record numbers; disabled flag or LLM failure → raw snippets used unchanged.
- **`src/fusion.py`:** stream B now renders `[W1]…[Wn]` with source labels and appends the derived summaries; header says "PubMed, NICE guidelines, web search".
- **`src/pipeline.py`:** each iteration log now stores `web_sources`, `web_summaries`, `search_failures` alongside `web_records`/`web_queries` — the new scanner outputs are auditable.
- **Config/env:** `SERPER_API_KEY` (your key, in `.env` — value never handled by the assistant), `SERPER_ENABLED`, `NICE_ENABLED`, `SCANNER_WEB_SUMMARY`, `SERPER_CACHE_DIR`; commented block in `.env.example`; `results/serper_cache/` gitignored. **No new dependencies** (`requests` already present).
- **Known effect:** Fusion's prompt text changed (source labels + summaries) → full-mode LLM cache entries from Phases 3–4 no longer match for cases that have hypotheses (expected; old cache files remain, results untouched).
- **Tests (36/36 PASS):** config/key detection (value never emitted) · Serper live search (3 records, shape, `source=web`) · cache written + second call served from disk in 0 ms · keyless path returns `serper_disabled_no_key` without raising · NICE live search (3 records, `nice.org.uk` URLs, level 1, year extracted) · NICE no-results and disabled paths · three-source `scan()` (12 records, all sources tagged, sorted level-1-first) · **simulated Serper outage → PubMed + NICE still return, failure recorded, `sources.serper=failed`** · web summaries produced/flag-off/disabled · full `run_case` (confidence, `model_trace`) · `py_compile` clean.
- **End-to-end proof (synthetic pneumonia vignette):** 1 iteration, confidence 0.8378, 1 hypothesis, **10 web records from all three sources** (`{pubmed: ok, nice: ok, serper: ok}`), 0 failures, 1 derived summary citing `[W8, …]`, 6 model-trace entries.
- **Demo replay:** `/api/cases/medqa_347` → 200, answer A.
- **Credit spend:** 4 Serper searches total (now cached); the free 2,500-credit tier is intact for Phase 7.

## Phase 6 delivered (2026-11-10)

- **Three named confidence factors (paper §II-A-4, `src/fusion.py`)** — the paper says
  the calibrated confidence "depends on factors such as the similarity score of the RAG,
  evidence score of web information, and degree of concordance between the two streams"
  but never publishes the formula. The four existing components are now **grouped onto
  exactly those three factors** (`rag_similarity` ← `retrieval_similarity` +
  `evidence_breadth`; `web_evidence_score` ← `web_support`;
  `cross_stream_concordance` ← `cross_source_agreement`) in a new `confidence_factors`
  block showing each factor's paper label, components, weights and contribution.
  **The grouping is presentational only** — the confidence is computed from the same
  left-to-right expression as before, so no published number moved and the optimizer's
  trigger decisions are unchanged (verified byte-identical in the test suite).
  Weights stay ours and `confidence_method` says so.
- **Per-diagnosis report rows (paper §II-A-4)** — `fuse()` now returns `diagnoses[]`,
  one deterministic row per candidate diagnosis with `evidence` (`[R#]` local passages,
  `[W#]` web records), `source` (`rag`/`web`/both), `confidence` (the same weights
  evaluated for that one diagnosis: base + 0.20·web + 0.20·concordance) and
  `above_threshold` at τ (paper §II-C: below-threshold recommendations are flagged, not
  silently served). No extra model call; the rows average to the aggregate confidence
  when condition names are unique. The rows ride in the report that the selection stage
  reads **and** in `log["report"]`, so the two can never drift apart.
- **`mode="web"` — the paper's web-only / "no RAG Analyzer" arm** (`src/pipeline.py`,
  `main.py --mode web`, `run_eval --modes`, `summarize_eval`): Clarifier → knowledge-based
  hypothesis proposal (`rag_analyzer.propose_without_retrieval`, 1 LLM call, falls back to
  the Clarifier's key concepts if the model fails) → Evidence Scanner (PubMed + NICE +
  Serper) → selection. **Single-pass by design**: no local index is touched (verified:
  `n_passages=0`, no retrieval trace) and, with one stream, there is nothing for Fusion to
  cross-check — symmetric with the `rag` arm. `confidence=None`, as in the other ablation
  arms. The paper never states how its web-only configuration forms per-hypothesis search
  terms once the RAG Analyzer is gone: that proposal call is our documented choice.
- **Optimizer feedback report (paper §II-A-5)** — `optimize()` now also returns
  `query_action ∈ {keywords, expand, reduce, rewrite}` (the paper's "keyword changes,
  query expansion/reduction"), normalised to `rewrite` if the model invents an action.
  The whole `feedback` dict is logged per iteration, and the pipeline comment documents
  how it reaches **both** downstream agents (refined_query → RAG Analyzer's
  `retrieval_query`, focus → the Clarifier `key_concepts` the Scanner reads).
  `optimizer.should_optimize()` (was dead code) is now the loop's single source of truth —
  same semantics, paper §II-D (τ or 3 iterations).
- **Mode guard** — an unknown `--mode` now raises instead of silently falling through to
  the full pipeline (which would have polluted an evaluation run).
- **Eval harness** — four configurations by default (`baseline,rag,web,full`); new
  per-question evidence columns (`n_passages`, `n_web_records`, `evidence_sources`);
  a per-question **evidence sidecar** `repl_eval_<dataset>_<stamp>_evidence.jsonl`
  (passages, web records with URL/source/level, Fig. 5 rungs, stop reasons, optimizer
  feedback, confidence factors, per-diagnosis rows, model trace; `--evidence 0` skips it);
  and every new run now uses the **`repl_` prefix** (closes gap-analysis item (b): new
  results can never be mixed up with the frozen prototype files). `summarize_eval.py`
  reads both prefixes and prints any mode present.
- **Tests (59/59 PASS):** three-factor mapping with byte-identical arithmetic
  (0.6547 old vs new), factor contributions summing to the confidence, per-diagnosis row
  schema/sources/evidence/threshold flags, row-mean ≈ aggregate, `should_optimize` truth
  table, `query_action` normalisation, web-hypothesis proposal + both fallbacks, unknown
  mode guard, CLI/harness wiring (`--help` greps), summarizer over a synthetic 4-mode CSV,
  a **live web-mode run** (13 external records from all three sources, no local index,
  correct option) and the **full-mode cache gate**.
- **Cache gate (the phase's acceptance criterion):** on the Phase-5 vignette, run 1 hit
  the Phase-5 cache for every prompt that did not change (clarifier, RAG Analyzer,
  scanner query, fusion — `cached=True`) and missed only `select` (the report now carries
  the per-diagnosis rows); an identical repeat run was **100% cache hits** and reproduced
  the same 0.8378 confidence.
- **Full-mode regression, same 20 IDs:** **18/20 (90%), 0 errors** — and *every*
  prediction is identical to the frozen prototype run (same two misses: `medqa_202` → C,
  `medqa_376` → E). Honest caveat: knobs differ (prototype `MAX_ITERS=2` +
  `SCANNER_LLM_QUERY=0` vs current defaults 3 / 1). Files:
  `results/repl_eval_medqa_20261110_001343.{csv,json}` + `_evidence.jsonl`.
  Transient NCBI 502/429 were recorded and degraded gracefully.
- **Cost:** ~150 flash-lite calls (mostly cache misses for prompts that legitimately
  changed) and **31 new Serper searches** (cache went 4 → 35 files) — the queries for
  the MedQA 20 are now warm, so the Phase 7 full run re-uses them for 0 credits.
- Known effect: the selection prompt changed (diagnoses block) → full-mode `select`
  cache entries from Phases 3–5 no longer match; old cache files remain untouched.

## Log

- **2026-11-09** — Phase 1 complete: full paper re-extraction (all 7 tables + Figs 1–6 OCR), full source audit (15 source files read), spec + gap analysis written. No source code modified in Phase 1.
- **2026-11-09** — Phase 2 complete: git checkpoint `9e507fd`; per-agent model configuration (shared Gemini default, per-agent provider/model overrides in `.env`, automatic Gemini fallback, per-call `model_trace`); 6/6 tests pass; docs (README, PROGRESS, `.env.example`) updated. No results/index/`.env` touched.
- **2026-11-09** — Phase 3 complete: staged Clarifier (deterministic normalisation → zero-shot NER/associations → ICD-10-CM via free NLM API → unchanged structured contract); new `src/coding.py` with transparent `matched_on`, cache, budget and graceful failure; 8/8 tests pass; docs updated. SNOMED CT remains deferred (licence). No results/index/`.env` touched.
- **2026-11-09** — Phase 4 complete: new `symptom_disease` collection (9,271 pts, existing collections untouched); τ=0.65 evidence threshold; term boosting (labeled approximation); Fig. 5 fallback ladder with per-rung trace + `retrieval_ms`; `length` metadata; 22/22 tests; before/after retrieval measurement recorded (0/20 → 0/20 lexical check, ladder fires 12/20). Thresholds from Table 2/Fig. 5 only — never tuned against eval questions. No results/`.env`/existing index touched.
- **2026-11-09** — Phase 5 complete: `src/serper.py` (key in `.env`, value never handled by the assistant; disk-cached; 401/402/429 → named reasons) + `src/nice.py` (keyless direct NICE search) + 3-source scanner with per-source failure isolation, `source` tags, extended evidence hierarchy and derived `[W#]` web summaries; iteration logs now record `web_sources`/`web_summaries`/`search_failures`; 36/36 tests + end-to-end vignette (10 records from all 3 sources, summary cited, conf 0.8378); demo replay 200; docs updated. Google Scholar deliberately skipped (no official API). 4 Serper credits spent, all cached. No results/index touched.
- **2026-11-10** — Phase 6 complete: confidence components mapped onto the paper's three named §II-A-4 factors (`confidence_factors`, arithmetic byte-identical, weights ours); per-diagnosis report rows (evidence `[R#]/[W#]`, source, confidence, τ-flag) added to the report the selection stage reads and to `log["report"]`; `mode="web"` ablation arm (Clarifier → knowledge-based hypothesis proposal → 3-source Scanner → selection; single-pass, no local index, no fusion) wired through `main.py`, `run_eval` (4 modes by default) and `summarize_eval`; optimizer feedback report gains `query_action` and `should_optimize()` now drives the loop; unknown modes raise; eval writes `repl_`-prefixed CSV/JSON + per-question evidence sidecar with new evidence columns. 59/59 tests; full-mode cache gate (untouched prompts still hit Phase-5 cache, only `select` missed, repeat run 100% cached at the same 0.8378); **full-mode regression on the same 20 IDs = 18/20 with every prediction identical to the prototype run** (`repl_eval_medqa_20261110_001343.*`). ~150 LLM calls, 31 new Serper searches (cache 4 → 35, warm for Phase 7). Frozen benchmark, index and `.env` untouched.
