# Clinical CDSS replication — adaptive multi-agent LLM + biomedical RAG

A two-person university AI-course research prototype.

> Öğdü, Ç. U., Arslanoğlu, K., Karaköse, M. *An Adaptive Multi-Agent LLM-Based Clinical
> Decision Support System Integrating Biomedical RAG and Web Intelligence.*
> IEEE Access, vol. 13, 2025. DOI: 10.1109/ACCESS.2025.3613340

Research/education prototype only. **Not a medical device. Do not use for patient care.**
The output is an experimental assessment with evidence traces, not a diagnosis.

---

## 1. Research objective

Implement a working prototype of the architecture proposed by the base paper: five
specialized agent roles that structure a clinical case, retrieve biomedical evidence
locally (RAG), fetch live literature from PubMed, fuse the two evidence streams, and
re-iterate when confidence is low — then measure it honestly on a fixed benchmark.

What this project **is not**:

- **Not** a reproduction of the paper's published accuracy (94.7% / 88% / 84%).
  Different model, smaller corpus, different protocol — we report our own numbers.
- **Not** a clinical product. No clinician review, no clinical validation.
- **Not** five independent LLMs — one Gemini model plays all five roles via distinct
  prompts (documented deviation).

## 2. What the prototype actually does

1. Accepts a clinical question (multiple-choice item or free text).
2. **Clarifier** turns the text into structured fields (symptoms, tests, concepts,
   a search query).
3. **RAG Analyzer** searches a local 34,206-chunk medical index with BioBERT embeddings
   and proposes hypotheses that must cite a retrieved passage.
4. **Evidence Scanner** searches PubMed live for each candidate condition and tags each
   record with an evidence level.
5. **Fusion** compares both streams, writes an agreement/conflict report, and computes
   a transparent confidence score.
6. If confidence < τ = 0.65, the **Optimizer** diagnoses the weak query and rewrites it;
   the loop runs again (up to 2 iterations in the evaluation, 3 by default).
7. A final selection prompt produces the answer + one-sentence rationale.
8. Every step is written to an audit log (passages with PMIDs, PubMed records, queries,
   per-iteration confidence components, stop reason, optimizer feedback).

It runs as a command line tool and as a small local web page (live analysis + saved
replay).

## 3. System architecture

```
question
   │
   ▼
[1 Clarifier] ──► structured case JSON + retrieval_query
   │
   ▼
[2 RAG Analyzer] ──► BioBERT embed ──► Qdrant (2 collections, RRF) ──► top-5 passages
   │                                   └──► hypotheses, each citing a passage
   ▼
[3 Evidence Scanner] ──► PubMed E-utilities ──► records + evidence levels
   │
   ▼
[4 Clinical Fusion] ──► agreement/conflict report
   │                   └──► confidence = 0.35·sim + 0.25·breadth + 0.20·web + 0.20·agreement
   │
   ├── confidence ≥ 0.65 ──────────────────────────────┐
   └── confidence < 0.65 and iterations remain          │
         │                                              │
         ▼                                              │
   [5 Optimizer] ──► reason + refined_query ──► back to 2,3,4
                                                                  │
                                                                  ▼
                                                    [final selection prompt] ──► answer
```

`src/pipeline.py` sequences the agents explicitly in Python (no CrewAI). The three
evaluation modes share the same code path:

| Mode | Path |
|---|---|
| `baseline` | final selection only — no retrieval, no agents |
| `rag` | local retrieval → formatted passages → final selection |
| `full` | Clarifier → RAG → Scanner → Fusion → Optimizer loop → final selection |

## 4. The five agent roles

All five are prompt-defined roles executed by **one** Gemini model. Each role has its
own module, prompt, and JSON output contract.

| # | Role | Module | What it does | Inputs → outputs |
|---|---|---|---|---|
| 1 | Clinical Text Clarifier | `src/clarifier.py` | **Staged pipeline (paper §II-A-1):** deterministic normalisation (NFKC + clinical abbreviation expansion, e.g. HTN→hypertension) → zero-shot NER into the paper's four classes (Symptom/Disease/Medication/Test Result) + association extraction → **ICD-10-CM mapping via the free NLM API** (`src/coding.py`, SNOMED deferred) → structured JSON: symptoms, duration, medications, test results, age/sex, key concepts, missing information, entities, associations, codes, 8–15 word retrieval query | question → clinical JSON + entities + codes |
| 2 | Symptom RAG Analyzer | `src/rag_analyzer.py` | Embeds the query, retrieves top-5 passages, proposes ≤3 hypotheses; **a hypothesis without a cited passage is dropped** | clinical JSON → passages + hypotheses + uncertainties |
| 3 | Evidence-Based Scanner | `src/evidence_scanner.py` | Builds one PubMed query per hypothesis (LLM-built when `SCANNER_LLM_QUERY=1` — current default; rule-based with `SCANNER_LLM_QUERY=0`), fetches records, tags evidence level (meta-analysis > RCT > cohort > case report) | hypotheses → PubMed records |
| 4 | Clinical Data Fusion | `src/fusion.py` | LLM compares both streams (agreements, disagreements, unsupported claims, gaps, inline `[R#]`/`[W#]` citations) **and** computes confidence from four logged numeric components | both streams → report + confidence |
| 5 | Adaptive Optimizer | `src/optimizer.py` | Runs only when confidence < τ. Must state *why* evidence was insufficient, then produce a meaningfully different query — never a blind retry | rejected fusion → reason + refined_query |

**How they communicate:** `pipeline.py` passes plain Python dicts between modules
(clarifier output → retrieval → scanner → fusion → optimizer → back around the loop).
There is no message bus and no shared memory; each agent call is a stateless Gemini
API request with its role prompt. Responses are cached on disk
(`results/llm_cache/`), so identical prompts replay instantly and deterministically.

**Why one model (initially):** a single Google AI Studio key was available, so role
separation is achieved through prompts rather than model selection. The paper used
different models per role (DeepSeek-R1, Gemini-2.0-Flash, Qwen2-72B, GPT-4o-mini).
This is deviation #1.

**Per-agent model configuration (replication Phase 2):** every stage resolves its own
`provider` + `model` from `config.AGENT_CONFIG` (`src/config.py`), defaulting to the
shared Gemini model. Any agent can be redirected to a different provider later without
code changes, for example in `.env`:

```powershell
RAG_ANALYZER_PROVIDER=groq          # only this agent changes
RAG_ANALYZER_MODEL=llama-3.3-70b-versatile
GROQ_API_KEY=...                    # keys only in .env, never committed
```

- Agents: `CLARIFIER`, `RAG_ANALYZER`, `EVIDENCE_SCANNER`, `FUSION`, `OPTIMIZER`,
  `SELECT` (the internal final-answer stage). Shared default via `LLM_PROVIDER`.
- Non-Gemini providers are called through OpenAI-compatible chat endpoints
  (openai, groq, openrouter, together, mistral, deepseek, xai, cerebras, perplexity —
  or any other name plus `<PROVIDER>_BASE_URL`).
- If a provider fails (missing key, auth error, timeout, unknown model), that agent
  **automatically falls back to Gemini** and the reason is logged — one
  `GEMINI_API_KEY` is always enough to run the whole system. Five providers are
  never required.
- Every execution log now records **`model_trace`**: one entry per LLM call with the
  agent, requested vs. actually executed provider/model, cache-hit status, fallback
  flag + reason, and duration. `parameters.agent_config` records the configuration
  a run started with.

## 5. Biomedical RAG implementation

| Aspect | Value |
|---|---|
| Embedding model | `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb` (identical model id to the paper) |
| Vector size / metric | 768-d, cosine |
| Vector store | Qdrant, embedded/local mode (single process) |
| Chunking (PubMed) | 512 tokens, 64 overlap (as reported in the paper) |
| Chunking (textbooks) | native paragraph segmentation (passage-sized already; documented deviation) |
| Collections | `biomedical` (11,244) + `textbooks` (13,691) + `symptom_disease` (9,271) |
| Fusion | Reciprocal Rank Fusion (RRF, k=60) across all collections |
| Quota | at least the best passage from each collection, so Fusion always sees independent streams |
| Similarity threshold τ | 0.65 for evidence selection (`RETRIEVAL_THRESHOLD`), enforced client-side |
| Term boosting | Active (Table 2) — documented approximation in `_boost_query()`: append absent priority terms, repeat present ones once, never boost "no …" negations, max 5 terms / 400 chars |
| Fallback (Fig. 5) | boosted search @0.65 → threshold reduction (0.55, 0.45) → simplify query @0.45 → remove symptoms @0.45 → no-threshold safety return (`search_with_fallback`) |
| top-k | 5 |

BioBERT is the **retriever only** — it converts text to vectors. It never writes the
answer; the Gemini model does that from the retrieved evidence.

## 6. Knowledge base and dataset sources

**Retrieval corpus (34,206 chunks):**

- 2,784 PubMed abstracts fetched from NCBI E-utilities (30 clinical topics) → 11,244
  chunks at 512/64.
- 13,691 textbook passages sampled from the public
  [`MedRAG/textbooks`](https://huggingface.co/datasets/MedRAG/textbooks) corpus
  (125,847 passages in 18 standard books: Harrison's Internal Medicine, Robbins
  Pathology, Nelson Pediatrics, Katzung Pharmacology, etc.). Up to 780 passages per
  book, random, seed 42 — a balanced coverage sample, **not** hand-picked for the exam
  questions. The base paper names no individual book titles, so its corpus cannot be
  copied; this public, citable substitute is deviation #3.
- 9,271 passages from the public Symptom-Disease Dataset
  ([`dux-tecblic/symptom-disease-dataset`](https://huggingface.co/datasets/dux-tecblic/symptom-disease-dataset),
  paper ref [30]): 7,043 patient-style narratives across 1,082 disease classes,
  passage = `"<narrative> Condition: <disease>"`, split 512/64. Built into its **own**
  collection by `scripts/build_symptom_index.py`, which never touches the existing
  two (it refuses to overwrite a non-empty collection without `--recreate`).

**Benchmark datasets (frozen, seed 42, 50 questions each):**

| Dataset | Source | File | Status |
|---|---|---|---|
| MedQA | `bigbio/med_qa` (test split) | `data/samples/medqa_sample.jsonl` | **evaluated** (first 20 questions) |
| PubMedQA | `qiaojin/PubMedQA` `pqa_labeled` (train) | `data/samples/pubmedqa_sample.jsonl` | frozen, **not evaluated** |
| MedBullets | `LangAGI-Lab/medbullets` (test) | `data/samples/medbullets_sample.jsonl` | frozen, **not evaluated** |

Question IDs are stable (`medqa_107` always means the same question). Benchmark answer
keys are used **only** by the scorer — they are never passed into the pipeline.

## 7. External evidence integration (Scanner)

The paper's Scanner sends **programmatic searches per hypothesis** through "a web
search API like **Serper**" to PubMed, Google Scholar and settled guideline databases
like **NICE** (§II-A-3). We implement all of that except Scholar:

| Source | Client | Auth | Evidence level |
|---|---|---|---|
| PubMed | `src/pubmed.py` — NCBI E-utilities (`esearch`+`efetch`) | none (free) | from publication types: guideline/meta-analysis 1 … case reports 4 |
| **NICE guidelines** | `src/nice.py` — direct `nice.org.uk/search` | **none (keyless)** | NICE guideline / quality standard / technology appraisal → 1; evidence summary → 2; other → 3 |
| **Web (Serper)** | `src/serper.py` — `google.serper.dev/search` | `SERPER_API_KEY` in `.env` | PubMed/PMC link → 3; any other web page → 4 (below case reports, design unverifiable) |
| Google Scholar | *not implemented* | — | no official API; scraping violates ToS (documented deviation) |

Behaviour:

- **Failure isolation** — each source is wrapped independently: a dead/expired Serper
  key, a rate limit or a NICE layout change degrades to fewer sources and is recorded
  in `search_failures` + `sources` (`{pubmed, nice, serper}` status per case). Evidence
  is never fabricated on failure.
- **Keyless by default** — PubMed and NICE need no key at all, so the whole system runs
  with only `GEMINI_API_KEY`; Serper simply reports `disabled_no_key`.
- **Web summaries** (paper: *"derives text summaries pertinent to each hypothesis"*) —
  one LLM pass turns the returned NICE/web pages into a 2–3 sentence summary per
  condition, citing `[W#]` record numbers. `SCANNER_WEB_SUMMARY=0` disables it; if the
  call fails, the raw snippets are used unchanged.
- **Caching** — Serper responses are disk-cached in `results/serper_cache/`
  (gitignored): identical queries cost 0 credits and reruns return the same pages.
- Polite throttles: PubMed 0.35 s/call, NICE 0.6 s/call, Serper 0.4 s/call.
- Queries are rule-based by default: the condition name alone, relevance-sorted
  (`SCANNER_LLM_QUERY=0`), with a retry/fallback if nothing is found. Long boolean
  keyword strings return nothing in PubMed — this was a real bug we fixed.
- Records are sorted strongest-first and carry `source`, `evidence_level`,
  `evidence_label`, `weight = 1/level`, `condition` and `query_used`.

## 8. Confidence-based adaptive optimization

```
confidence = 0.35 × retrieval_similarity      (best BioBERT cosine of returned passages)
           + 0.25 × evidence_breadth          (fraction of top-k slots actually filled)
           + 0.20 × web_support               (fraction of hypotheses supported by a web record)
           + 0.20 × cross_source_agreement    (overlap between RAG hypotheses and web-supported ones)
```

- **Mapped onto the paper's three named factors (§II-A-4)** — the paper says the
  calibrated confidence "depends on factors such as the similarity score of the RAG,
  evidence score of web information, and degree of concordance between the two streams"
  but never publishes the formula, so the same four components are grouped as:

  | Paper factor (§II-A-4) | our components | weights (ours) |
  |---|---|---|
  | `rag_similarity` — similarity score of the RAG | `retrieval_similarity`, `evidence_breadth` | 0.35 + 0.25 |
  | `web_evidence_score` — evidence score of web information | `web_support` | 0.20 |
  | `cross_stream_concordance` — concordance between the two streams | `cross_source_agreement` | 0.20 |

  The grouping is presentational: **not one number changed** (the sum is the same
  left-to-right expression as before). `confidence_factors` in every log shows each
  factor's components, weights and contribution.
- **Per-diagnosis report rows (§II-A-4)** — the report now carries, for every
  candidate diagnosis, its `evidence` (`[R#]` local passages, `[W#]` web records),
  `source` (`rag` / `web` / both), `confidence` (same weights, evaluated for that one
  diagnosis) and an `above_threshold` flag at τ. Deterministic — no extra model call.
- Threshold τ = 0.65 (as reported in the paper).
- All four components **and** the three factors are logged separately for every
  iteration, so any score can be audited component-by-component.
- **The score is a heuristic, not a calibrated probability.** A confidence of 0.85 does
  not mean "85% likely correct" — both of our full-pipeline errors scored > 0.84.
  The paper publishes the 0.65 threshold but never its formula; these weights are ours
  (deviation #7).

## 9. Technology stack

| Layer | Choice |
|---|---|
| Language | Python 3.11 (`.python-version` pins 3.11) |
| LLM | Gemini REST API via `requests` (direct calls, explicit timeouts) — model `gemini-flash-lite-latest` from `.env` |
| Embeddings | `sentence-transformers` loading BioBERT, CPU-only (~11–16 chunks/s, ~30 s model load) |
| Vector DB | `qdrant-client`, embedded mode |
| Data | `datasets` (Hugging Face) for benchmarks/textbooks, `requests` for PubMed |
| Orchestration | explicit Python (`src/pipeline.py`) — no CrewAI, no framework |
| Web demo | Python stdlib `http.server` + single static page (`web/index.html`) |
| Config/secrets | `.env` via `python-dotenv` (never committed) |
| Eval | `csv`/`json` outputs, 4-worker `ThreadPoolExecutor` |

## 10. Evaluation methodology

- **Dataset:** first 20 questions of the frozen MedQA sample (seed 42, IDs fixed).
- **Configurations:** `baseline`, `rag`, `web`, `full` — the paper's four arms
  (no retrieval / local RAG only / external evidence only / fused + feedback loop).
  The two ablation arms are single-pass by design: with one stream there is nothing
  for Fusion to cross-check. Same fixed IDs and gold answers in every mode.
- **Scoring:** exact match on the option letter. (Free-text datasets would use the
  paper's cosine ≥ θ = 0.80 rule — labelled lenient — but none have been evaluated.)
- **Settings:** `SCANNER_LLM_QUERY=0`, `MAX_ITERS=2`, `TOP_K=5`, 4 workers, temperature 0,
  τ = 0.65. Recorded in the JSON as `run_config` together with the dataset SHA-256,
  model, embedding model, and Python version.
- **Failure handling:** an errored run is recorded with its error text and counted as
  wrong — never silently dropped or imputed.
- **Gold isolation:** `run_case()` receives only the question and options; the gold
  answer is read by the scorer after the fact.

## 11. Actual verified MedQA results (20 questions, run 2026-10-26 22:33)

| Configuration | Correct | Accuracy | Errors | Mean confidence | Mean iterations |
|---|---|---|---|---|---|
| Baseline (LLM only) | 17/20 | 85.0% | 0 | — | 0 |
| RAG only | 17/20 | 85.0% | 0 | — | 1 |
| Full (RAG + PubMed + fusion + loop) | **18/20** | **90.0%** | 0 | 0.791 | 1.5 |

- **Paired vs baseline:** 17 both correct, 0 baseline-only, 1 full-only
  (`medqa_107`, fixed after 2 iterations), 2 neither (`medqa_202`, `medqa_376`).
- **RAG alone changed no decision** — same 17/20, same three errors as baseline.
- **Both full-mode errors were high-confidence** (`202` at 0.855, `376` at 0.849):
  confidence measures cross-source agreement, not correctness.
- **+1/20 is not statistically significant.** Directional result only.
- Files: `results/eval_medqa_20261026_223318.csv` / `.json`. An earlier run
  (`..._214410`) had three Qdrant concurrency failures in RAG mode; the client-init
  race was fixed and that run is superseded, not merged.
- Latency (first uncached run): 28.8 s baseline, ~32 s RAG, 144.1 s full per case.
  Cached reruns take ~13 s/case — **not** deployment latency.

### Phase 6 replication regression (2026-11-10, `full`, same 20 IDs)

| Configuration | Correct | Accuracy | Errors | Mean confidence | Mean iterations |
|---|---|---|---|---|---|
| Full (Phase 3–6 code) | **18/20** | **90.0%** | 0 | 0.7666 | 1.35 |

- **Every prediction is identical to the prototype run** — the same two misses
  (`medqa_202` → C, `medqa_376` → E) and the same 18 correct answers, so the
  Phase 3–6 changes (staged Clarifier, τ-ladder retrieval, three-source scanner,
  factor mapping, per-diagnosis report) moved no decision on this set.
- Honest caveat: the settings are *not* identical — the frozen prototype run used
  `MAX_ITERS=2` + `SCANNER_LLM_QUERY=0`, this regression used the current defaults
  `MAX_ITERS=3` + `SCANNER_LLM_QUERY=1`. Same IDs, same gold, different knobs.
- **Cache gate:** prompts untouched since Phase 5 still hit the Phase-5 cache
  (clarifier / RAG Analyzer / scanner query / fusion all `cached=True`); only the
  selection prompt missed, because the report now carries the per-diagnosis rows.
  An identical repeat run was 100% cache hits and reproduced the same confidence.
- Files: `results/repl_eval_medqa_20261110_001343.{csv,json}` plus
  `..._evidence.jsonl` (per-question evidence sidecar: passages, web records with
  URLs, Fig. 5 rungs, stop reasons, feedback, model trace).
- Transient NCBI 502/429 responses were recorded and degraded gracefully — 0 errors;
  one case (`medqa_2`, porcelain gallbladder) returned 0 external records in all
  three iterations and still answered correctly from the local RAG stream.

## 12. Project status — implemented / tested / evaluated / future

| Category | Items |
|---|---|
| **Implemented** (exists in code) | Five agents + pipeline loop; staged Clarifier (deterministic normalisation → zero-shot NER/associations → ICD-10-CM via free NLM API); BioBERT/Qdrant three-collection RRF retrieval with τ=0.65 evidence threshold, term boosting and the Fig. 5 fallback ladder; **three-source evidence scanner (PubMed + keyless NICE + Serper) with per-source failure isolation, source tagging and derived web summaries**; confidence heuristic with logged components **mapped onto the paper's three named §II-A-4 factors** + **per-diagnosis report rows (evidence/source/confidence/τ-flag)**; optimizer feedback report (`reason`/`focus`/`query_action`/`refined_query`); **`web`-only ablation mode** (no local index, no fusion); CLI + local web demo (live analysis and saved replay); audit logging; LLM disk cache + Serper response cache; evaluation harness with `repl_` outputs + per-question evidence sidecar; benchmark sampler; per-agent provider/model configuration with automatic Gemini fallback and per-agent `model_trace` logging |
| **Tested** (verified technically) | End-to-end demos `medqa_347` and `medqa_107`; Qdrant concurrency stress test (8 threads / 16 calls, 0 failures, counts unchanged); retrieval smoke tests; Phase 4 suite (22 checks); Phase 5 suite (36 checks: Serper live+cache+keyless, NICE keyless parse, three-source scan, outage isolation, web summaries, full `run_case`); **Phase 6 suite (59 checks: three-factor mapping, byte-identical confidence arithmetic, per-diagnosis rows, `should_optimize`, `query_action`, web-mode hypothesis fallback, mode guard, CLI/harness wiring, live web run, full-mode cache gate)**; JSON parsing/LLM retry paths; gold-answer isolation; demo replay viewer |
| **Evaluated** (measured experimentally) | MedQA, 20 questions × 3 configurations (prototype run, §11); **Phase 6 full-mode regression on the same 20 IDs: 18/20, identical predictions, 0 errors** (§11) |
| **Not yet evaluated** | PubMedQA (50 frozen) and MedBullets (50 frozen) — prepared but **no results**; free-text θ = 0.80 scoring path (implemented, never exercised on a benchmark); 50-question MedQA runs |
| **Future work** (outside current scope) | Larger evaluation, retrieval-quality metric with stemming, confidence calibration study, two-model role heterogeneity (needs a second API key), expert/clinician review, proactive background scanning (paper's "continuous" refresh) |

## 13. Known deviations from the paper

| Paper | This repo | Status |
|---|---|---|
| 5 different LLMs for the 5 roles | one shared Gemini model by default; per-agent provider/model overrides via `.env`, Gemini fallback, per-call `model_trace` (replication Phase 2) | **deviation** — architecture ready for multi-model, one key still suffices |
| Gemini-2.0-Flash | `gemini-flash-lite-latest` | **forced** — 2.0/2.5 retired, 3.8-flash 503/timeouts; only reliable model on our key |
| CrewAI orchestration | explicit Python pipeline | **deviation** (transparent, testable) |
| Serper + Scholar + NICE + 1,000 PDFs | **Serper web search (key in `.env`) + NICE (keyless, direct) + PubMed E-utilities** + 18 MedRAG textbooks + the paper's own Symptom-Disease dataset (ref [30]) | **partial (Phase 5)** — Google Scholar skipped (no official API, ToS), PDF/OCR pipeline skipped (decision) |
| Textbooks/guidelines/epidemiological datasets (titles unpublished) | balanced 780/book sample from MedRAG | **deviation** (paper's corpus unreproducible) |
| Term boosting algorithm (Table 2: "Active", weights unpublished) | documented approximation: append absent priority terms, repeat present ones once, never boost "no …" negations, max 5 terms / 400 chars (`_boost_query`) | **approximation** (Phase 4) — paper gives no formula |
| SNOMED CT / ICD-10 mapping | staged Clarifier + **ICD-10-CM via free NLM Clinical Tables API** (real codes only, `matched_on` recorded; SNOMED CT deferred — licence) | **partial** (Phase 3) — never claims codes we did not return |
| Confidence formula | documented heuristic, **grouped onto the paper's three named §II-A-4 factors** (`rag_similarity` / `web_evidence_score` / `cross_stream_concordance`) with per-diagnosis rows; arithmetic unchanged | **not published in paper** — factors named by the paper, weights ours |
| max 3 feedback cycles | default 3; evaluation used 2 (time budget) | documented |
| 1,000 PDFs → 200k chunks → 18M embeddings | 34,206 chunks | paper's arithmetic is inconsistent; not treated as a target |

Additional measurement limitations:

1. **n=20** — one question separates the configurations; not significant.
2. **Free-text scoring (θ = 0.80) is lenient** vs standard benchmark protocols.
3. **Live PubMed is not cached** — full-mode iteration counts/confidence can shift
   between runs even though LLM responses are cached (accuracy held at 18/20 across
   both runs).
4. **MedQA items are exam stems**, not patient narratives — the Clarifier sees a
   question stem (the paper has the same property).
5. **A crude lexical check** finds the gold answer string in the top-5 for 0/20 MedQA
   items; it has no stemming and misses paraphrases, so it understates retrieval —
   reported as a limitation, not a metric.

## 14. Installation

```powershell
cd "E:\D Drive\clinical-cdss"
pip install -r requirements.txt
```

Create `.env` from the template and put your Google AI Studio key in it:

```powershell
copy .env.example .env
notepad .env
```

```
GEMINI_API_KEY=...        <- your key; never commit or share it
GEMINI_MODEL=gemini-flash-lite-latest
QDRANT_URL=               <- blank = use the local index (recommended)
QDRANT_API_KEY=
```

Never commit `.env` (already in `.gitignore`). Never paste the key into chat, logs,
slides, or commands.

## 15. Building the knowledge base (no API key needed)

```powershell
python scripts\fetch_pubmed.py          # ~2,800 abstracts from NCBI (free)
python scripts\build_index.py           # chunk -> BioBERT -> Qdrant (biomedical)
python scripts\build_textbook_index.py  # MedRAG textbooks -> Qdrant (textbooks)
python scripts\build_symptom_index.py   # HF Symptom-Disease [30] -> Qdrant (symptom_disease)
                                        #   creates ONLY that collection; refuses to
                                        #   overwrite a non-empty one without --recreate
python scripts\eval_retrieval.py 20     # retrieval-only diagnostic (no LLM)
python scripts\smoke_retrieval.py       # inspect retrieved passages
python scripts\sample_datasets.py       # freeze 50 questions/dataset, seed 42
```

The built index lives in `data/raw/qdrant_index/` (git-ignored; copy the whole project
folder, not just the GitHub clone). Qdrant embedded mode allows **one process at a
time** — close other Python windows if you see a storage-lock error.

## 16. Running the program

```powershell
python main.py --case medqa_347                    # full pipeline on one benchmark item
python main.py --case medqa_107                    # adaptive-correction item (2 iterations)
python main.py --question "45M, fever, cough..."   # free-text case
python main.py --question "..." --mode rag         # retrieval only
python main.py --question "..." --mode web         # external evidence only (ablation arm)
python scripts\show_case.py medqa_107              # pretty-print the saved audit log
python scripts\summarize_eval.py                   # results table + paired comparison
```

## 17. Demo instructions

Full script with expected output: **`docs/DEMO.md`**. Short version:

```powershell
# Demo A — normal single-pass workflow (answer A, conf 0.8577, 1 iteration)
python main.py --case medqa_347
python scripts\show_case.py medqa_347

# Demo B — adaptive correction (0.4499 -> optimizer -> 0.8277, answer A, 2 iterations)
python main.py --case medqa_107
python scripts\show_case.py medqa_107
```

Web demo (live analysis + saved replay of the two cases):

```powershell
python web_demo.py
# open http://127.0.0.1:8765
```

Replays read `results/execution_logs/` and make **no** API calls; live analysis calls
Gemini + PubMed and uses the local index. Lab-PC setup: `LAB_PC_DEMO_SETUP.txt`.

Demo commands finish in seconds because LLM responses are cached; retrieval and PubMed
are still real. First uncached runs took ~100 s per case.

## 18. Evaluation

```powershell
$env:SCANNER_LLM_QUERY='0'; $env:MAX_ITERS='2'
python evaluation\run_eval.py --dataset medqa --n 20 --modes baseline,rag,web,full --workers 4
python scripts\summarize_eval.py
```

New runs are always written with the **`repl_` prefix** so they can never be confused
with the frozen prototype benchmark: `results/repl_eval_medqa_<timestamp>.csv` + `.json`
(per-case rows + `run_config` with dataset SHA-256, model, thresholds, Python version)
and `results/repl_eval_medqa_<timestamp>_evidence.jsonl` — one line per (question, mode)
with the passages, web records (title/URL/source/level), Fig. 5 fallback rungs, stop
reasons, optimizer feedback, confidence factors, per-diagnosis rows and the per-agent
model trace. `--evidence 0` skips the sidecar.

PubMedQA/MedBullets runs are **optional experiments requiring team approval** — the
same command with `--dataset pubmedqa` or `--dataset medbullets`.

## 19. Reproducibility

- Every LLM call is cached in `results/llm_cache/` keyed by model + prompt +
  temperature + JSON mode; all 173 entries are from `gemini-flash-lite-latest`.
  Reruns of identical prompts are free and deterministic.
- Every case writes an audit log to `results/execution_logs/`: structured input,
  retrieved passages with PMIDs/URLs, PubMed records, hypotheses, web queries,
  per-iteration confidence components, optimizer reason, stop reason, final selection.
- Benchmark IDs are frozen in `data/samples/` — never resample between runs.
- Live PubMed is not cached: full mode is not bit-for-bit reproducible across days.

## 20. Project directory structure

```
clinical-cdss/
├── main.py                    # CLI entry point
├── web_demo.py                # local web demo server (live + replay)
├── web/index.html             # demo page
├── .env                       # secrets (git-ignored)
├── .env.example               # config template
├── requirements.txt
├── src/
│   ├── config.py              # all thresholds, model ids, paths
│   ├── pipeline.py            # five-agent orchestration + loop
│   ├── clarifier.py           # agent 1
│   ├── rag_analyzer.py        # agent 2
│   ├── evidence_scanner.py    # agent 3
│   ├── fusion.py              # agent 4 (+ confidence formula)
│   ├── optimizer.py           # agent 5
│   ├── retrieval.py           # BioBERT + Qdrant + RRF + fallback
│   ├── pubmed.py              # NCBI E-utilities client
│   └── llm.py                 # Gemini REST client, retries, disk cache
├── evaluation/run_eval.py     # 3-mode benchmark harness
├── scripts/                   # corpus build, sampling, demo/report helpers
├── data/
│   ├── samples/*.jsonl        # frozen benchmark questions (50 × 3 datasets)
│   └── raw/qdrant_index/      # 34,206-vector local index (git-ignored)
├── results/
│   ├── eval_medqa_20261026_223318.{csv,json}   # the reported run
│   ├── execution_logs/        # per-case audit logs (incl. medqa_347, medqa_107)
│   └── llm_cache/             # cached LLM responses (git-ignored)
├── docs/
│   ├── SLIDES.md              # 7-slide milestone deck
│   ├── DEMO.md                # demo script + expected output
│   ├── METHOD_NOTES.md        # protocol, settings, weaknesses
│   ├── PROGRESS.md            # status + measured results
│   └── RELATED_WORK.md        # paper analysis + positioning
├── PRESENTATION_QA.txt / PROGRESS_PRESENTATION_SLIDES.txt /
│   PROGRESS_PRESENTATION_SPEECH.txt    # 12-slide progress presentation materials
└── LAB_PC_DEMO_SETUP.txt      # lab PC setup checklist
```

## 21. Citation

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
