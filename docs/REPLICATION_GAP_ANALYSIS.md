# Replication Gap Analysis — paper requirement vs. actual code

**Basis:** `docs/REPLICATION_SPECIFICATION.md` (paper extraction) compared against **source code** (not README claims). Each entry names the exact files involved.
**Date of audit:** 2026-11-09 · Repository state: 2 commits (`f9ad858`, `82505b9`) + working-tree modifications (README, docs, `.env.example`, 2 demo logs) — **commit or back up before Phase 2 changes.**

## Classification legend

| Class | Meaning |
|---|---|
| **FULLY IMPLEMENTED** | Code does what the paper requires (minor cosmetic differences ok) |
| **PARTIALLY IMPLEMENTED** | Core behavior exists, but specific disclosed requirements are missing or differ |
| **NOT IMPLEMENTED** | Requirement is absent from the code |
| **CANNOT BE REPRODUCED** | The paper does not publish enough information to replicate this exactly |

## Summary

| # | Requirement | Class |
|---|---|---|
| G-01 | Five-agent architecture & information flow | FULLY IMPLEMENTED |
| G-02 | Original model assignments (4 LLMs) + per-agent model config | PARTIALLY IMPLEMENTED (per-agent override architecture ✅ done in Phase 2; the paper's 4-model lineup intentionally not used — documented deviation) |
| G-03 | CrewAI orchestration | NOT IMPLEMENTED (behavioral equivalent in place; **decision confirmed 2026-11-09: keep Python orchestration**, documented deviation) |
| G-04 | Clinical text processing (normalization → NER → association) | **IMPLEMENTED (Phase 3)** — deterministic normalisation + explicit entities/associations schema (NER & structuring share one zero-shot pass — paper doesn't state call counts) |
| G-05 | SNOMED CT / ICD-10 mapping | **PARTIALLY IMPLEMENTED (Phase 3)** — ICD-10-CM live via free NLM API (real codes only, `matched_on` recorded); SNOMED CT deferred (licence) |
| G-06 | Biomedical corpus construction (1,000 PDFs, OCR, filters) | **PARTIALLY IMPLEMENTED (Phase 4)** — third corpus source (paper ref [30] Symptom-Disease) added as its own collection; PDF/OCR pipeline skipped (decision); exact corpus CANNOT BE REPRODUCED |
| G-07 | BioBERT embeddings (768-D) | FULLY IMPLEMENTED |
| G-08 | Chunking & indexing (512/64, 500/80 fallback, page metadata) | **PARTIALLY IMPLEMENTED (Phase 4)** — `length` metadata ✅, 500/80 + page numbers impossible without PDFs |
| G-09 | Qdrant cosine retrieval, top-k | FULLY IMPLEMENTED (Phase 4: τ=0.65 enforced) |
| G-10 | Three-level fallback (Fig. 5 strategies) | **IMPLEMENTED (Phase 4)** — full Fig. 5 ladder + per-rung trace; safety return + per-step timing are ours |
| G-11 | Medical term boosting | **IMPLEMENTED (Phase 4, APPROXIMATION)** — paper algorithm unpublished; `_boost_query` documented as our interpretation |
| G-12 | External evidence sources (Serper / Scholar / NICE) | PARTIALLY IMPLEMENTED (PubMed only) |
| G-13 | Evidence classification & weighting | FULLY IMPLEMENTED (for disclosed rules) |
| G-14 | Clinical fusion (cross-check, conflict, report) | PARTIALLY IMPLEMENTED |
| G-15 | Confidence formula | CANNOT BE REPRODUCED (ours is a labeled heuristic) |
| G-16 | Adaptive optimization mechanics (τ, refine, re-query, stop) | FULLY IMPLEMENTED |
| G-17 | Reflexive learning from case similarities | CANNOT BE REPRODUCED (mechanism unpublished) |
| G-18 | Web interface (annotated report, trace, logs) | PARTIALLY IMPLEMENTED |
| G-19 | Datasets MedQA / PubMedQA / MedBullets (n=50, scoring) | PARTIALLY IMPLEMENTED (samples ✓; PubMedQA/MedBullets runs not executed; MedBullets scoring differs from paper) |
| G-20 | Baseline configuration | PARTIALLY IMPLEMENTED (our own plain-LLM arm; paper ran none) |
| G-21 | RAG-only and web-only experiments | RAG-only FULLY · web-only NOT IMPLEMENTED |
| G-22 | Iteration experiment (Table 6) | NOT IMPLEMENTED |
| G-23 | Agent ablation experiment (Table 7) | NOT IMPLEMENTED |
| G-24 | Evaluation metrics (micro P/R/F1, specificity, Table 5) | NOT IMPLEMENTED |
| G-25 | Published results (94/88/84 …) | reported-only — never claim reproduction |
| G-26 | Evaluation harness requirements (hashes, records, isolation) | PARTIALLY IMPLEMENTED |

---

## Detailed entries

### G-01 · Five-agent architecture and information flow — **FULLY IMPLEMENTED**
- **Paper requires (§II-A, Fig. 2):** 5 separate agents; Clarifier → (RAG Analyzer ∥ Evidence Scanner) → Fusion → Optimizer → feedback loop.
- **Our code:** `src/clarifier.py`, `src/rag_analyzer.py`, `src/evidence_scanner.py`, `src/fusion.py`, `src/optimizer.py` — five distinct modules, each with its own prompt, input contract and output contract; `src/pipeline.py::run_case` runs them sequentially in exactly Fig. 2 order, with the Optimizer feeding refined queries back into iterations.
- **Missing:** none. **Files:** — **Deps:** — **Feasible:** done.

### G-02 · Model assignments + per-agent provider/model configuration — **PARTIALLY IMPLEMENTED** (architecture ✅ Phase 2, 2026-11-09)
- **Paper requires (§II intro, §II-A-1…5):** DeepSeek-R1 (Clarifier), Gemini-2.0-Flash (RAG Analyzer + Optimizer), Qwen2-72B (Scanner), GPT-4o-mini (Fusion); no fine-tuning.
- **Our code:** `src/llm.py` + `src/config.py` — **Phase 2 delivered the full override architecture:** `AGENT_CONFIG` (per-agent `<AGENT>_PROVIDER`/`<AGENT>_MODEL` in `.env`, shared `LLM_PROVIDER`/Gemini default); provider dispatch (native Gemini + OpenAI-compatible adapter for openai/groq/openrouter/together/mistral/deepseek/xai/cerebras/perplexity or any name + `<PROVIDER>_BASE_URL`); **automatic fallback to the Gemini chain** on any provider failure with the reason logged; per-call thread-local **`model_trace`** recorded in every `run_case` log (`model_trace` + `parameters.agent_config`). All five agent modules and `pipeline._select` tag their calls with `agent=...`. Cache key format unchanged → existing cache hits preserved. `.env.example` documents the overrides (commented, no real keys).
- **Still deviating (intentional):** all five roles run on one Gemini model by default (`gemini-flash-lite-latest`) because only one reliable key is available; the paper's four-model lineup is not used. The architecture allows a later switch per agent without code changes. **Never require five providers:** a missing/failing optional provider always falls back to Gemini.
- **Missing:** nothing for the project's Phase-2 requirement. **Files:** done (`config.py`, `llm.py`, 5 agents, `pipeline.py`, `.env.example`). **External deps:** none new.
- **Tests (2026-11-09):** config smoke ✓, broken-provider→Gemini fallback trace ✓, cached full `run_case` model_trace ✓, scanner tag ✓, demo replay ✓, `py_compile` ✓.

### G-03 · CrewAI orchestration — **NOT IMPLEMENTED** (behavioral equivalent in place)
- **Paper requires (§II-A):** CrewAI framework; "Agent, Crew, Process, Task" modules; sequential process + dynamic feedback loops.
- **Our code:** `src/pipeline.py` — explicit Python orchestration of the same sequential flow + loop.
- **Missing:** CrewAI itself. **Files:** would be a new orchestration layer replacing `pipeline.py` wiring.
- **External deps:** `crewai` (heavy transitive dependency tree, version churn).
- **Feasible:** technically yes, but high regression risk for zero behavioral gain. **Recommendation: keep Python orchestration, document as deviation** (pending user confirmation).

### G-04 · Clinical text processing (staged) — **IMPLEMENTED (Phase 3, 2026-11-09)**
- **Paper requires (§II-A-1):** sequential stages: Text Normalization (noise/abbrev/spelling) → NER with classes *Symptom, Disease, Medication, Test Result* → Association Extraction (entity-to-entity context, e.g. symptom↔anatomical region) → ontology mapping → JSON.
- **Our code (Phase 3):** `src/clarifier.py` now runs the stages explicitly:
  1. **Normalisation — deterministic** (`normalize_text`): NFKC unicode cleanup, whitespace collapse, ~70 high-precision clinical abbreviation expansions with word boundaries (HTN→hypertension, SOB→shortness of breath, y/o→years old, …). The normalised text is what the model receives and it is stored in `normalized_text`.
  2. **NER + association extraction — one zero-shot LLM pass** with an explicit staged schema: `entities[] {text, type ∈ {Symptom, Disease, Medication, Test Result}}` (plus an honest `Other` bucket if the model insists on a fifth class) and `associations[] {entity_a, relation, entity_b}` covering the paper's "numbness ↔ left arm" pattern. Outputs are sanitised/validated in code (`_clean_entities`, `_clean_associations`).
  3. **Terminology mapping** → see G-05.
  4. **Structured output** — the original JSON contract (`symptoms`, `duration`, `severity`, `medications`, `test_results`, `patient_age`, `patient_sex`, `key_concepts`, `missing_information`, `retrieval_query`) is byte-for-byte preserved; the new fields are additive, so every downstream agent kept working unchanged (verified by the Phase 3 pipeline test).
- **Deviation (documented):** stages 2 and 4 share the model's single zero-shot pass — the paper never states how many model calls its Clarifier makes, so call count is not falsifiable either way; normalisation and coding are genuinely deterministic stages outside the model.
- **Files:** `src/clarifier.py` (done). **External deps:** none (no spaCy/NLTK). **Tests:** normalisation unit ✓, entity/association cleaning ✓, staged `clarify()` on a synthetic narrative (9 entities across all four classes, 6 associations) ✓, full-pipeline downstream ✓, disabled-path ✓, demo replay ✓.

### G-05 · SNOMED CT / ICD-10 mapping — **PARTIALLY IMPLEMENTED (Phase 3: ICD-10-CM live; SNOMED deferred)**
- **Paper requires (§II-A-1):** map every identified clinical concept to SNOMED CT or ICD-10 codes. No method/version disclosed.
- **Our code (Phase 3):** new `src/coding.py` maps Disease/Symptom entities to **real ICD-10-CM codes** through the free NLM Clinical Tables API (`https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search`, `sf=code,name`, no key): per-case disk cache, 0.2 s throttle, per-case query budget (`ICD10_MAX_QUERIES`, default 8), local re-rank (description-starts-with → token overlap → shorter description) plus progressive token fallback so "community acquired pneumonia" resolves to **J18.9 with `matched_on: "pneumonia"` recorded transparently**. Every code entry carries `system: "ICD-10-CM"` and is only emitted if NLM actually returned it. Medication/Test Result entities are listed in `unmapped_concepts` with the honest reason that ICD-10-CM does not code drugs/tests; `coding_status` ∈ {ok, degraded, disabled}. `ICD10_ENABLED=0` disables the API cleanly. SNOMED CT is never claimed.
- **Missing:** SNOMED CT mapping (licence/membership required → deferred by decision 2026-11-09). Medication coding would need RxNorm, test coding LOINC — both free via NLM but out of the paper's stated scope ("SNOMED CT or ICD-10").
- **Files:** `src/coding.py` (new), `src/clarifier.py`, `src/config.py` (`ICD10_ENABLED`, `ICD10_MAX_QUERIES`), `.env.example`, `.gitignore` (cache file).
- **External deps / licensing:** NLM API — free, no key, no cost. **SNOMED CT** — license/membership required → stays deferred.
- **Honesty rule kept: we claim terminology mapping only for codes actually returned.** Tests: exact match (R06.02) ✓, fallback match (J18.9) ✓, no-match and budget reasons ✓, disabled path ✓, full pipeline retains codes ✓.

### G-06 · Biomedical corpus construction — **PARTIALLY IMPLEMENTED** (exact corpus **CANNOT BE REPRODUCED**)
- **Paper requires (§II-B-1, Fig. 3):** >15-keyword queries + study filters + ~20 excluded topics → ≤1,000 parallel PDF downloads (relevance-ranked, Entrez history) → PyMuPDF extraction (65 pg/s) → 28% OCR + ICU normalization → regex section stripping → segmentation → cleaning → embedding → Qdrant; plus HF Symptom-Disease Dataset [30]; claims 200k chunks / 18M embeddings / 2.1 GB / 38 min (and separately "889 articles + 5,630 patient texts").
- **Our code:** `scripts/fetch_pubmed.py` (30 fixed topics, `retmax=100`, E-utilities, → 2,784 abstracts), `scripts/build_index.py` (512/64 chunking → `biomedical` collection, 11,244 points), `scripts/build_textbook_index.py` (MedRAG/textbooks, 18 books × 780 passages → `textbooks` collection, 13,691 points), **`scripts/build_symptom_index.py` (Phase 4: paper ref [30] `dux-tecblic/symptom-disease-dataset`, 7,043 narratives → `symptom_disease` collection, 9,271 points — creates only its own collection, refuses to overwrite a non-empty one without `--recreate`)**. Total **34,206 chunks**.
- **Missing vs paper:** PDF full-text pipeline (PyMuPDF), OCR (28% of pages), ICU normalization, regex section filtering, >15-keyword query construction with exclusions, 1,000-PDF scale, page-number metadata, 500/80 page-fallback chunking.
- **Files:** `scripts/fetch_pubmed.py`, `scripts/build_index.py`, `scripts/build_textbook_index.py`, `scripts/build_symptom_index.py`, `src/config.py` (`COLLECTIONS`).
- **External deps/datasets:** PMC open-access PDFs (free), PyMuPDF+OCR (free, already installed system-wide) → ref [30] = *D. Tecblic, Symptom-Disease Dataset, HuggingFace* (**added Phase 4**, free, cached under `data/raw/symptom_disease/`).
- **Feasible:** exact paper corpus **not feasible** (query recipe, topic list and index never released; paper contradicts itself on size). Closest defensible alternative **now built** (all three corpus sources); optional upgrade: the 1,000-PDF OCR pipeline (needs approval either way).
- **Recommendation:** keep abstracts/textbooks/symptom-disease; skip the 1,000-PDF OCR pipeline (multi-day detour, paper's corpus still unreproducible).

### G-07 · BioBERT embeddings — **FULLY IMPLEMENTED**
- **Paper requires:** `BioBERT-mnli-snli-scinli-scitail-mednli-stsb`, 768-D, frozen weights, offline indexing.
- **Our code:** `src/config.py:23-25` (exact model id), `src/retrieval.py:18-26,49-58` (SentenceTransformer, CPU, normalized vectors), `scripts/build_index.py:73-79` (Qdrant COSINE 768-D, offline one-off build). **Missing:** none.

### G-08 · Chunking and indexing — **PARTIALLY IMPLEMENTED**
- **Paper requires (Table 2):** Recursive Character Text Splitter 512/64; **page-based fallback 500/80** when segments inadequate; drop too-brief/label-only segments; metadata = source filename, **page number**, text length.
- **Our code:** `scripts/build_index.py:15-39` separator-based splitter (recursive-style) at `CHUNK_SIZE=512 / CHUNK_OVERLAP=64` (`config.py:31-32`), drops chunks ≤60 chars; payload = title/url/pmid/source/topic/pubtypes/year/chunk_index. **Phase 4: every hit now carries `length`** (`retrieval.py`, computed from the payload text) and the new `symptom_disease` collection stores `length` per chunk.
- **Missing:** 500/80 fallback path (meaningless without page-based segments); page numbers (only possible with the PDF pipeline, G-06).
- **Files:** `scripts/build_index.py`, `scripts/build_symptom_index.py`, `src/config.py`, `src/retrieval.py`.
- **Feasible:** yes (length done; page numbers only with PDFs). **Index rebuilds must preserve existing collections** (see safety note below).

### G-09 · Qdrant retrieval — **FULLY IMPLEMENTED**
- **Paper requires:** cosine similarity, ANN top-k, offline precomputed index, parametric threshold/max-results.
- **Our code:** `src/retrieval.py` (Qdrant COSINE, `query_points`, `TOP_K=5` — paper never states k; local embedded mode or cloud URL via `.env`), `search()`/`search_with_fallback()` expose parametric k and an optional `score_threshold`. **Phase 4: the paper's τ=0.65 is enforced client-side** (`RETRIEVAL_THRESHOLD`, over-fetch ×5 then filter — works in embedded and cloud mode). Missing: none.

### G-10 · Three-level fallback (Figure 5) — **IMPLEMENTED (Phase 4)**
- **Paper requires (Fig. 5):** on unsatisfactory results apply **Strategy 1 Threshold Reduction → Strategy 2 Simplify Query → Strategy 3 Remove Symptoms**; explicitly not top-k rotation; decision-trace metadata (source, doc ID, page, fragment length, retrieval time).
- **Our code:** `src/retrieval.py::search_with_fallback` now walks the published ladder: boosted search @τ=0.65 → threshold reduction @0.55 → @0.45 → simplify query @0.45 → remove symptoms @0.45 → **no-threshold safety return** (our addition so downstream stages never receive zero passages). Widen-k (the old prototype behaviour) was removed — it is not in Fig. 5. Every rung appends `{level, strategy, threshold, n_hits, best_score, query_used, ms}` to `trace["levels"]`, plus `retrieval_ms` total; full-mode runs store it per iteration (`pipeline.py` → `iterations[i].retrieval_trace`). Hit metadata: chunk_id/score/rrf/collections/text/title/url/source/topic/pmid/**length** — no page numbers (G-06).
- **Implemented:** all three strategies + correct Fig. 5 order + fixed docstring + per-step timing. **Missing:** page numbers (impossible without PDFs); per-*hit* timing (per-step timing instead — the paper does not define where the clock stops).
- **Files:** `src/retrieval.py`, `src/rag_analyzer.py` (supplies boost/symptom/simplified terms), `src/config.py` (`RETRIEVAL_THRESHOLD`, `TERM_BOOST`).
- **Feasible:** done, no external deps.
- **Note:** the 0.65/0.55/0.45 rungs come from Table 2 + Fig. 5 annotations and were **not** tuned against evaluation questions (`scripts/eval_retrieval.py` runs are measurement/reporting only).

### G-11 · Medical term boosting — **IMPLEMENTED (Phase 4, APPROXIMATION)**
- **Paper requires (§II-B-2, Table 2, Table 7):** repeat high-priority symptom terms in the query so they gain weight in the embedding ("shortness of breath", "chest pain" score higher); **Active (baseline)**; ablation → 80.0%.
- **Our code:** `src/retrieval.py::_boost_query` (called from `search_with_fallback` when `TERM_BOOST=1`): appends absent priority terms, repeats present ones once, **never boosts negated terms** ("no …", "denies", "without", "negative for", "not "), max 5 terms / 400 chars. Priority terms come from `src/rag_analyzer.py` (`key_concepts + symptoms`). `TERM_BOOST` config flag exists — **required for the Table-7 "term boosting off" ablation arm**.
- **Missing:** nothing functional. **The paper never specifies the mechanism** (term repetition? query concatenation? per-term score multiplier?) → `_boost_query` is explicitly labeled an **our-interpretation approximation** in code, README §13 and METHOD_NOTES.
- **Files:** `src/retrieval.py`, `src/rag_analyzer.py`, `src/config.py`, `.env.example`.
- **Feasible:** done. No deps.

### G-12 · External evidence sources — **PARTIALLY IMPLEMENTED**
- **Paper requires (§II-A-3):** Serper web-search API → PubMed, Google Scholar, NICE guideline databases; programmatic per-hypothesis queries; proactive continuous scanning.
- **Our code:** `src/pubmed.py` (NCBI E-utilities esearch/efetch, 0.35 s throttle, never fabricates evidence on failure) used by `src/evidence_scanner.py` (per-hypothesis LLM query when `SCANNER_LLM_QUERY=1`, rule-based condition query otherwise, retry fallback). **On-demand per case, not proactive background scanning.**
- **Missing:** Serper integration, Google Scholar, NICE guidelines, background refresh.
- **Files to modify:** new `src/serper.py` (+ `.env` key), `src/evidence_scanner.py` (multi-source fan-out, source labels), `src/config.py`, `.env.example`.
- **External deps & cost:** **Serper.dev** — free tier (2,500 searches, credit card optional?) → *user must register and put `SERPER_API_KEY=` in `.env` themselves (never in chat)*. **Google Scholar has no official API** — scraping violates ToS → recommend NOT implementing. **NICE** — public website; direct fetch of specific guidance pages is possible without a key (free), scope-limited.
- **Feasible:** PubMed ✓ (done) · Serper ✓ after user gets free key · NICE ✓ simple fetch, optional · Scholar ✗ (policy risk).
- **Robustness requirement:** unavailable sources must never crash the pipeline — current `pubmed.search()` already returns `[]` on failure; same pattern must wrap Serper/NICE.

### G-13 · Evidence classification and weighting — **FULLY IMPLEMENTED** (for disclosed rules)
- **Paper requires (§II-A-3):** rank systematic reviews/meta-analyses/guidelines above case reports; no numeric weights disclosed.
- **Our code:** `src/pubmed.py:91-110` `LEVEL_BY_PUBTYPE` (systematic review/meta-analysis/guideline=1, RCT/review/clinical trial=2, observational=3, case reports=4) + `weight = 1/level` (`evidence_scanner.py:79`), sorted best-first (`:83`). Matches every disclosed ordering rule.
- **Missing:** nothing disclosable (numeric weights CANNOT BE REPRODUCED — paper gives none). Source-credibility scoring is paper's own future work (§IV).

### G-14 · Clinical fusion — **PARTIALLY IMPLEMENTED**
- **Paper requires (§II-A-4, §II-C):** cross-validation of RAG vs web streams; conflict-detection protocol; calibrated confidence from RAG similarity + web evidence score + cross-stream concordance; semantic fusion into "best data"; structured report with evidence/source/confidence per diagnosis.
- **Our code:** `src/fusion.py` — LLM extracts agreements/disagreements/unsupported claims/coverage/evidence gaps/summary (conflict detection ✓, cross-check ✓); deterministic components `retrieval_similarity`, `evidence_breadth`, `web_support` (token-overlap ≥0.5 rule, `:73-86`), `cross_source_agreement` (Jaccard-style, `:89-96`); selection stage in `pipeline.py::_select`.
- **Missing:** exact paper calibration (see G-15); below-threshold *withholding* behavior (paper §II-C says sub-threshold data "not made available"; we instead trigger re-query — closer to §II-D; document choice).
- **Files:** `src/fusion.py` (possible component alignment), `src/pipeline.py`.
- **Feasible:** yes.

### G-15 · Confidence formula — **CANNOT BE REPRODUCED**
- **Paper requires:** τ=0.65; confidence depends on RAG similarity, web evidence strength, cross-source concordance. **Formula/weights never published.**
- **Our code:** `src/fusion.py:129-140` — `0.35·sim + 0.25·evidence_breadth + 0.20·web_support + 0.20·agreement`, explicitly labeled "replication heuristic", logged component-by-component.
- **Missing:** the real formula (unobtainable). **Option (needs approval — changes pipeline behavior/numbers):** re-shape to the paper's three named factors only (drop or fold in `evidence_breadth`) so the component names mirror §II-A-4. Weights remain ours either way.
- **Files:** `src/fusion.py`, docs. **Feasible:** approximation only; honest labeling required.

### G-16 · Adaptive optimization mechanics — **FULLY IMPLEMENTED**
- **Paper requires (§II-D, §II-A-5):** τ=0.65; feedback = structured recommendations (not blind retry); restructure failed queries; alternative queries for missing symptoms; guidance to **both** RAG and Scanner; stop at τ or max 3 iterations.
- **Our code:** `src/optimizer.py` (diagnose-why + refined query + focus terms, anti-repetition rules), `src/pipeline.py:128-166` (loop, `stop_reason` = threshold_met / iteration_limit / low_confidence_requery, `MAX_ITERS=3`, `TAU=0.65`), refined query + focus injected into `clinical` for the next RAG/Scanner pass (`:162-166`). Scanner queries derive from the hypotheses → indirectly guided ✓.
- **Missing:** nothing disclosed. (Cross-case learning = G-17.)

### G-17 · Reflexive behavior / learning from case similarities — **CANNOT BE REPRODUCED**
- **Paper requires (§II-D bullet):** "exhibit reflexive behavior that optimizes itself by learning from case similarities over time". **No mechanism, data structure or algorithm is described anywhere.**
- **Our code:** none (per-case only).
- **Feasible:** would require inventing a memory/learning design → out of scope unless user explicitly requests an approximation; classify as future work in docs.

### G-18 · Web interface — **PARTIALLY IMPLEMENTED**
- **Paper requires (§III, Fig. 6):** single-page interface + REST APIs; free-text input; annotated report appending **source used, potential sources of variation, calibrated confidence**; iteration trace; auditable logs; synthetic de-identified cases; **role-based access**; ~2–4 min/case.
- **Our code:** `web_demo.py` (stdlib HTTP server; `POST /api/analyze` → job → `src/pipeline.run_case` **live, not fabricated**; `GET /api/cases/<id>` replays saved audit logs; `GET /api/jobs/<id>`), `web/index.html` (free-text input, structured summary, RAG+PubMed evidence panels with scores/levels, confidence meter with τ, per-iteration trace with stop reasons, final annotated answer+rationale, "Research prototype / Not validated for clinical use / confidence is not a probability" labeling), `main.py` CLI.
- **Missing:** role-based access control (single local user — optional, document as deviation); explicit "potential sources of variation" field (partially covered by confidence components + rationale); Fig. 6-style **per-hypothesis confidence tiers (High/Mod/Low)** in the report; source list on the final report (per-run sources are shown per iteration, not aggregated).
- **Files:** `web_demo.py`, `web/index.html`, `src/pipeline.py` (aggregate sources/variations into final log).
- **Feasible:** yes, no deps.

### G-19 · Datasets and scoring — **PARTIALLY IMPLEMENTED**
- **Paper requires (Table 3, §III):** MedQA 50 (A–E), PubMedQA 50 free-text, MedBullets 50 free-text; exact-match for MCQ; cosine ≥ θ=0.80 for free-text; **MedBullets scored as free-text per §III**.
- **Our code:** `scripts/sample_datasets.py` (seed 42, frozen IDs; `bigbio/med_qa` 5-option, `qiaojin/PubMedQA pqa_labeled`, `LangAGI-Lab/medbullets` — MedBullets stored **with options**), `evaluation/run_eval.py` (MCQ exact match; free-text cosine ≥ `THETA=0.80` via `_cos`), `data/samples/*.jsonl` (50/50/50 frozen, sha256 recorded at run time).
- **Missing / differs:** (a) **MedBullets executed as MCQ exact-match, but the paper scores it free-text + cosine** → to follow the paper: answer MedBullets as free-text (ignore options in the prompt) and score with θ; we can additionally record exact-match for transparency. (b) **PubMedQA and MedBullets runs never executed** (only MedQA n=20 × 3 modes exists). (c) Paper's exact question IDs **CANNOT BE REPRODUCED** (never published) — ours are seeded and frozen, which is the defensible substitute.
- **Files:** `scripts/sample_datasets.py` (no re-freeze! keep current IDs), `evaluation/run_eval.py` (free-text branch for MedBullets), run commands.
- **Feasible:** yes; runs cost API calls (see plan).

### G-20 · Baseline configuration — **PARTIALLY IMPLEMENTED**
- **Paper requires:** "baselines" = Table 4 literature models (cited, not re-run). The paper ran **no plain-LLM arm of its own system**.
- **Our code:** `pipeline.py:99-107` `mode="baseline"` (LLM only, no retrieval) — our own arm, analogous to Table 4's "No agent" category.
- **Missing:** nothing runnable can be "reproduced" here (paper gave no internal baseline protocol). Keep our arm, documented as **our addition**, and never equate it with Table 4 numbers.
- **Files:** `evaluation/run_eval.py` (already supports `--modes baseline`).

### G-21 · RAG-only and web-only experiments — **RAG-only FULLY · web-only NOT IMPLEMENTED**
- **Paper requires:** RAG-only = no Scanner/Fusion (Table 7 "no Scanner"=RAG-only 70.0? — careful: §II-C names *RAG-only 65.0* and *web-only 70.0*, but Table 7 shows no-RAG-Analyzer=65.0 and no-Scanner=70.0; §II-C's labels are swapped relative to Table 7. Implement both arms as **agent-removal variants** (G-23) and report both labels with this caveat).
- **Our code:** `mode="rag"` exists (`pipeline.py:109-121`). **No web-only mode** (Clarifier → Scanner → Fusion with empty RAG stream → selection).
- **Missing:** `mode="web"` in `pipeline.py` + `--modes web` support (already generic) — must handle empty-RAG gracefully in `fusion.py::_fmt_rag` (returns "(no RAG evidence)" ✓ already).
- **Files:** `src/pipeline.py`, `evaluation/run_eval.py` (mode list docs).
- **Feasible:** yes, small.

### G-22 · Iteration experiment (Table 6) — **NOT IMPLEMENTED**
- **Paper requires:** 20 MedQA questions at iteration counts 1→5, accuracy + processing time per count (86.3→95.5%, 2.5→5.5 min); exact-match A–E; averages reported (± basis unstated).
- **Our code:** nothing sweeps iteration counts (`MAX_ITERS` is a fixed env value; `run_eval.py` has no sweep mode).
- **Missing:** a sweep script: for k in 1..5 → set `MAX_ITERS=k`, run the same 20 question IDs in `full` mode, record accuracy/latency per k, write one timestamped CSV/JSON.
- **Files:** new `scripts/run_iter_sweep.py` (+ reuse `run_eval.py` internals), `src/pipeline.py` (accept per-run max_iters override — currently read from config at call time, so pass-through parameter needed).
- **External deps:** none. **API cost:** 20 q × (1+2+3+4+5) ≈ 300 iteration-units ≈ 1,000–1,500 LLM calls (flash-lite, cached per prompt) → **ask approval before running**. Note paper's own 4–5 iterations exceed its max-3 rule; we mirror Table 6 for fidelity and flag the inconsistency.

### G-23 · Agent ablation experiment (Table 7) — **NOT IMPLEMENTED**
- **Paper requires:** remove one component at a time, same 20 MedQA questions: Clarifier 80.0, RAG Analyzer 65.0, Scanner 70.0, Fusion 60.0, Optimizer 70.0, Term Boosting 80.0.
- **Our code:** no ablation flags anywhere (grep-verified). Closest: 3-mode eval (baseline/rag/full).
- **Missing:** `--ablate {clarifier,rag,scanner,fusion,optimizer,term_boost}` wiring: pipeline must be able to skip/replace each agent (e.g., clarifier_off → pass raw question as clinical dict; rag_off → empty hypotheses but still scan; scanner_off → empty web; fusion_off → confidence from RAG only / skip conflict report; optimizer_off → hard stop after iteration 1 with no re-query; term_boost_off → needs G-11 flag first) + eval mode labels + same-20-ID guarantee.
- **Files:** `src/pipeline.py`, `src/retrieval.py` (boost flag), `evaluation/run_eval.py`, `src/config.py`.
- **Feasible:** yes, moderate work. **API cost:** 6 arms × 20 q ≈ 120 case-runs → **ask approval before running**.

### G-24 · Evaluation metrics (micro P/R/F1, specificity, Table 5) — **NOT IMPLEMENTED**
- **Paper requires (§III):** one-vs-all micro metrics (formulas fully published) + θ-cosine binary rule + Table 5 reporting for 3 datasets.
- **Our code:** `evaluation/run_eval.py` computes accuracy only (`:133-143`); grep confirms no precision/recall/F1/specificity anywhere. θ-rule implemented for free-text (`:58-61`) ✓.
- **Missing:** metric computation from TP/FP/TN/FN per the paper's reduction (MedQA: correct→TP1+TN4, wrong→FP1+FN1+TN3), specificity, Table-5-style summary block in JSON output; also recomputation for existing CSVs (gold+predicted columns exist → **free, offline**).
- **Files:** `evaluation/run_eval.py` (+ small `scripts/metrics_from_csv.py` for old files).
- **Feasible:** yes, no deps, no API cost.

### G-25 · Published results — **reported-only**
94/88/84, Table 4–7 values are the paper's. We reproduce **protocol**, never the numbers, unless our results genuinely support them. Our prototype result stays clearly separated: baseline 17/20, RAG 17/20, full 18/20 (MedQA n=20, 2026-10-26).

### G-26 · Evaluation harness requirements (project §6) — **PARTIALLY IMPLEMENTED**
- **Already done:** same frozen IDs across modes ✓ · gold isolated from prompts ✓ (gold letter/answer never sent) · per-question correctness ✓ · confidence + iteration counts ✓ · latency ✓ · error rows recorded, never dropped ✓ · timestamps ✓ · dataset sha256 + full run config in JSON ✓ (`run_eval.py:147-163`) · temperature 0 + prompt-hash LLM cache ✓.
- **Missing:** (a) **retrieval/evidence records per eval question** — `save_log=False` in eval (`:51`) so no per-question evidence is kept; add a sidecar `repl_eval_<stamp>_evidence.jsonl` (or opt-in logs). (b) **New results must be visually distinguishable from prototype results** — current naming `eval_medqa_<stamp>` risks mixing with the 2026-10-26 files; use prefix `repl_` (or `results/replication/`) for all new runs. (c) Dataset hash for *all three* sample files at run start (currently hashes only the selected dataset — fine, but keep). (d) Worker concurrency vs local Qdrant single-process: stress-tested 16/16 PASS, keep `--workers 4`.
- **Files:** `evaluation/run_eval.py`, `.gitignore` (whitelist new result files if they should be committed).
- **Feasible:** yes.

---

## Safety constraints baked into the plan

1. **Never touch:** `data/raw/qdrant_index/` (existing collections: 11,244 biomedical + 13,691 textbooks vectors; Phase 4 added a separate `symptom_disease` collection of 9,271), `results/eval_medqa_20261026_*.{csv,json}` (prototype benchmarks), `results/execution_logs/medqa_{107,347}.json`, `.env` API keys (only *add* new key lines with user approval).
2. **`build_index.py:74-75` deletes its collection on rerun** — any corpus work must target **new collection names** or run only after explicit approval.
3. New evaluation outputs use the `repl_` prefix; prototype outputs stay untouched.
4. Every phase ends with tests + a status report before the next phase starts.
5. Git: commit current working tree as a checkpoint before Phase 2 (command below).

## Phased implementation plan (for approval)

| Phase | Content | Main files | New deps / cost | Test gate |
|---|---|---|---|---|
| **1** ✅ | Paper audit → `REPLICATION_SPECIFICATION.md` + this document | docs | none | docs reviewed by team |
| **2** | Per-agent model configuration (shared Gemini default, per-agent overrides in `.env`, fallback, per-agent model logging) + git checkpoint | `config.py`, `llm.py`, 5 agents, `pipeline.py`, `.env.example` | none | unit smoke: each agent logs its model; existing demo replay unchanged |
| **3** | Clarifier: staged schema (entities with paper's 4 classes + associations) + ICD-10-CM coding via NLM free API (SNOMED deferred pending license decision) | `clarifier.py`, new `coding.py` | NLM API (free, no key) | schema validates; downstream agents still run; demo case replays |
| **4** | RAG: term boosting (labeled approximation), Fig-5 fallback order (threshold reduction → simplify → remove symptoms), docstring fix, chunk-length metadata, optional Symptom-Disease dataset as **new** collection | `retrieval.py`, `rag_analyzer.py`, `config.py`, new `scripts/fetch_symptoms.py` | HF dataset (free) | retrieval smoke tests + `eval_retrieval.py` comparison old vs new (old index untouched) |
| **5** | Scanner: Serper source (**only if you register a free key**), optional NICE page fetch, multi-source fan-out with per-source failure isolation | new `src/serper.py`, `evidence_scanner.py` | Serper free tier (user-registered) | scanner returns PubMed-only when key absent; no crash paths |
| **6** | Fusion/optimizer: components mirror paper's three named factors (weights stay ours, labeled), guidance report content review, `mode="web"` added | `fusion.py`, `optimizer.py`, `pipeline.py` | none | full-mode regression: same 20 IDs, cached prompts where unchanged |
| **7** | Evaluation: micro metrics (G-24), web-only mode (G-21), ablation flags (G-23), iteration sweep (G-22), MedBullets free-text scoring (G-19), evidence sidecar + `repl_` prefix (G-26) — **runs executed after your approval of API spend** | `run_eval.py`, new `scripts/run_iter_sweep.py`, `pipeline.py` | flash-lite API calls (est. ~1,500–2,500 calls total) | metric unit tests on synthetic rows; then n=20 MedQA canary |
| **8** | Web report: aggregate sources + "sources of variation", per-hypothesis High/Mod/Low tiers (Fig. 6), keep prototype labeling | `web_demo.py`, `web/index.html`, `pipeline.py` | none | manual demo + saved-case replay |
| **9** | Final: README rewrite, `REPLICATION_PROGRESS.md` final status, deviation ledger, honest results table | README, docs | none | docs match code (spot-check) |

## Decisions (answered 2026-11-09 before Phase 2 — recorded in `REPLICATION_PROGRESS.md`)

1. Plan approved ✅ (git checkpoint `9e507fd` executed before code changes).
2. CrewAI → **keep Python orchestration**, documented deviation.
3. External sources → **Serper + NICE** (user registers free Serper key → `.env`); Scholar skipped (no official API).
4. SNOMED → **deferred (license)**; **ICD-10-CM now** via free NLM API (Phase 3).
5. Corpus → **add HF Symptom-Disease dataset as a new collection**; skip 1,000-PDF/OCR pipeline (Phase 4).
6. Phase 7 API budget → **staged: ablation first (~700 calls), review, then separate approval** for the iteration sweep + PubMedQA/MedBullets runs.
