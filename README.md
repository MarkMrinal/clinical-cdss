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
3. **RAG Analyzer** searches a local 24,935-chunk medical index with BioBERT embeddings
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
| 1 | Clinical Text Clarifier | `src/clarifier.py` | Structures free text: symptoms, duration, medications, test results, age/sex, key concepts, missing information, and an 8–15 word retrieval query | question → clinical JSON |
| 2 | Symptom RAG Analyzer | `src/rag_analyzer.py` | Embeds the query, retrieves top-5 passages, proposes ≤3 hypotheses; **a hypothesis without a cited passage is dropped** | clinical JSON → passages + hypotheses + uncertainties |
| 3 | Evidence-Based Scanner | `src/evidence_scanner.py` | Builds one PubMed query per hypothesis (rule-based by default, `SCANNER_LLM_QUERY=0`; LLM-built when enabled), fetches records, tags evidence level (meta-analysis > RCT > cohort > case report) | hypotheses → PubMed records |
| 4 | Clinical Data Fusion | `src/fusion.py` | LLM compares both streams (agreements, disagreements, unsupported claims, gaps, inline `[R#]`/`[W#]` citations) **and** computes confidence from four logged numeric components | both streams → report + confidence |
| 5 | Adaptive Optimizer | `src/optimizer.py` | Runs only when confidence < τ. Must state *why* evidence was insufficient, then produce a meaningfully different query — never a blind retry | rejected fusion → reason + refined_query |

**How they communicate:** `pipeline.py` passes plain Python dicts between modules
(clarifier output → retrieval → scanner → fusion → optimizer → back around the loop).
There is no message bus and no shared memory; each agent call is a stateless Gemini
API request with its role prompt. Responses are cached on disk
(`results/llm_cache/`), so identical prompts replay instantly and deterministically.

**Why one model:** a single Google AI Studio key was available, so role separation is
achieved through prompts rather than model selection. The paper used different models
per role (DeepSeek-R1, Gemini-2.0-Flash, Qwen2-72B, GPT-4o-mini). This is deviation #1.

## 5. Biomedical RAG implementation

| Aspect | Value |
|---|---|
| Embedding model | `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb` (identical model id to the paper) |
| Vector size / metric | 768-d, cosine |
| Vector store | Qdrant, embedded/local mode (single process) |
| Chunking (PubMed) | 512 tokens, 64 overlap (as reported in the paper) |
| Chunking (textbooks) | native paragraph segmentation (passage-sized already; documented deviation) |
| Collections | `biomedical` (11,244) + `textbooks` (13,691) |
| Fusion | Reciprocal Rank Fusion (RRF, k=60) across both collections |
| Quota | at least the best passage from each collection, so Fusion always sees two streams |
| Fallback | 3 levels: exact → widen k → simplified query (`search_with_fallback`) |
| top-k | 5 |

BioBERT is the **retriever only** — it converts text to vectors. It never writes the
answer; the Gemini model does that from the retrieved evidence.

## 6. Knowledge base and dataset sources

**Retrieval corpus (24,935 chunks):**

- 2,784 PubMed abstracts fetched from NCBI E-utilities (30 clinical topics) → 11,244
  chunks at 512/64.
- 13,691 textbook passages sampled from the public
  [`MedRAG/textbooks`](https://huggingface.co/datasets/MedRAG/textbooks) corpus
  (125,847 passages in 18 standard books: Harrison's Internal Medicine, Robbins
  Pathology, Nelson Pediatrics, Katzung Pharmacology, etc.). Up to 780 passages per
  book, random, seed 42 — a balanced coverage sample, **not** hand-picked for the exam
  questions. The base paper names no individual book titles, so its corpus cannot be
  copied; this public, citable substitute is deviation #3.

**Benchmark datasets (frozen, seed 42, 50 questions each):**

| Dataset | Source | File | Status |
|---|---|---|---|
| MedQA | `bigbio/med_qa` (test split) | `data/samples/medqa_sample.jsonl` | **evaluated** (first 20 questions) |
| PubMedQA | `qiaojin/PubMedQA` `pqa_labeled` (train) | `data/samples/pubmedqa_sample.jsonl` | frozen, **not evaluated** |
| MedBullets | `LangAGI-Lab/medbullets` (test) | `data/samples/medbullets_sample.jsonl` | frozen, **not evaluated** |

Question IDs are stable (`medqa_107` always means the same question). Benchmark answer
keys are used **only** by the scorer — they are never passed into the pipeline.

## 7. PubMed integration (external evidence)

- Client: `src/pubmed.py` — NCBI E-utilities (`esearch` + `efetch`), free, no API key.
- Polite throttle: one call per 0.35 s (NCBI allows ~3/s unauthenticated).
- Each record is tagged with an evidence level from its publication types and weighted
  accordingly; records are sorted strongest-first.
- Queries are rule-based by default: the condition name alone, relevance-sorted
  (`SCANNER_LLM_QUERY=0`), with a retry/fallback if nothing is found. Long boolean
  keyword strings return nothing in PubMed — this was a real bug we fixed.
- The paper used Serper + Google Scholar + NICE guidelines; we use PubMed only
  (free, citable, stable) — deviation #4.

## 8. Confidence-based adaptive optimization

```
confidence = 0.35 × retrieval_similarity      (best BioBERT cosine of returned passages)
           + 0.25 × evidence_breadth          (fraction of top-k slots actually filled)
           + 0.20 × web_support               (fraction of hypotheses supported by a PubMed record)
           + 0.20 × cross_source_agreement    (overlap between RAG hypotheses and web-supported ones)
```

- Threshold τ = 0.65 (as reported in the paper).
- All four components are logged separately for every iteration, so any score can be
  audited component-by-component.
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
- **Configurations:** `baseline`, `rag`, `full` — same 20 questions in all three
  (verified: identical IDs and gold answers per mode).
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

## 12. Project status — implemented / tested / evaluated / future

| Category | Items |
|---|---|
| **Implemented** (exists in code) | Five agents + pipeline loop; BioBERT/Qdrant dual-collection RRF retrieval with 3-level fallback; PubMed scanner with evidence levels; confidence heuristic with logged components; optimizer re-query; CLI + local web demo (live analysis and saved replay); audit logging; LLM disk cache; evaluation harness; benchmark sampler |
| **Tested** (verified technically) | End-to-end demos `medqa_347` and `medqa_107`; Qdrant concurrency stress test (8 threads / 16 calls, 0 failures, counts unchanged); retrieval smoke tests; JSON parsing/LLM retry paths; gold-answer isolation; demo replay viewer |
| **Evaluated** (measured experimentally) | MedQA, 20 questions × 3 configurations (results in §11) |
| **Not yet evaluated** | PubMedQA (50 frozen) and MedBullets (50 frozen) — prepared but **no results**; free-text θ = 0.80 scoring path (implemented, never exercised on a benchmark); 50-question MedQA runs |
| **Future work** (outside current scope) | Larger evaluation, retrieval-quality metric with stemming, confidence calibration study, two-model role heterogeneity (needs a second API key), SNOMED CT / ICD-10 coding, expert/clinician review, guidelines beyond PubMed |

## 13. Known deviations from the paper

| Paper | This repo | Status |
|---|---|---|
| 5 different LLMs for the 5 roles | one Gemini model, role-specific prompts | **deviation** (single API key) |
| Gemini-2.0-Flash | `gemini-flash-lite-latest` | **forced** — 2.0/2.5 retired, 3.8-flash 503/timeouts; only reliable model on our key |
| CrewAI orchestration | explicit Python pipeline | **deviation** (transparent, testable) |
| Serper + Scholar + NICE + 1,000 PDFs | PubMed E-utilities + 18 MedRAG textbooks | **deviation** (free, citable) |
| Textbooks/guidelines/epidemiological datasets (titles unpublished) | balanced 780/book sample from MedRAG | **deviation** (paper's corpus unreproducible) |
| SNOMED CT / ICD-10 mapping | LLM extraction only | **not implemented** (licence) |
| Confidence formula | documented heuristic | **not published in paper** |
| max 3 feedback cycles | default 3; evaluation used 2 (time budget) | documented |
| 1,000 PDFs → 200k chunks → 18M embeddings | 24,935 chunks | paper's arithmetic is inconsistent; not treated as a target |

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
python evaluation\run_eval.py --dataset medqa --n 20 --modes baseline,rag,full --workers 4
python scripts\summarize_eval.py
```

Outputs `results/eval_medqa_<timestamp>.csv` + `.json` with per-case rows, a summary,
and a `run_config` block (dataset SHA-256, model, thresholds, Python version).
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
│   └── raw/qdrant_index/      # 24,935-vector local index (git-ignored)
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
