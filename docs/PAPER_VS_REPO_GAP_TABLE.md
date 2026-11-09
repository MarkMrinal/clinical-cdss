# Paper ↔ Repository Gap Table

**Paper:** Öğdü, Arslanoğlu, Karaköse — *"An Adaptive Multi-Agent LLM-Based Clinical Decision Support System Integrating Biomedical RAG and Web Intelligence"*, IEEE Access vol. 13, 2025. DOI `10.1109/ACCESS.2025.3613340`

**Purpose:** section-by-section comparison of what the paper specifies vs. what this repository actually implements, so the team can decide *systematically* which gaps (if any) to close.

**Sources of truth**
- Paper text: `paper_full.txt` (all 15 pages). Tables 1–7 have **no text layer** (vector outlines) → extracted verbatim via OCR (`page3/9/11/12/13_ocr.txt`).
- Repo: verified against source files (paths below), not from memory.

**The 7 columns:** ① Paper specifies ② Repo location ③ Our implementation ④ Differs from paper ⑤ Missing vs paper ⑥ Required to reproduce ⑦ Inputs publicly available?

**Status marks:** ✅ matches · ⚠️ differs (documented deviation) · ❌ missing · 📄 reported-only (paper claim we do **not** reproduce)

> **Reminder:** we never claim reproduction of 94.7 % / 88 % / 84 %. Our honest result is full 90 % vs baseline 85 % vs RAG 85 % on MedQA n=20 (+1 case, not significant).

---

## A. Section I — Introduction & positioning (Table 1)

| ID | ① Paper specifies | ② Repo location | ③ Our implementation | ④ Differs | ⑤ Missing | ⑥ Required to reproduce | ⑦ Public? |
|---|---|---|---|---|---|---|---|
| A1 | Table 1: *This Study* = Multi-Agent + external info + dynamic web access + multi-source fusion (vs Traditional LLMs / KG4Diagnosis / Medical Agents) | `README.md`, `docs/RELATED_WORK.md` | All four *This Study* capabilities present: 5 roles, BioBERT RAG, PubMed web stream, fusion | — (positioning table only) | — | n/a | n/a |
| A2 | Contribution 1: multi-agent system that "optimizes itself over time" incl. *"learning from case similarities"* (§II-D) | `src/optimizer.py`, `src/pipeline.py` | Within-case re-query loop only | no cross-case learning | reflexive cross-case learning | mechanism **never specified** in paper | ❌ (unpublished) |
| A3 | Corpus claims: intro says **889 articles + 5,630 patient texts**; §II-B-1 says **1,000 PDFs → 200,000 chunks → 18M embeddings**, 2.1 GB, 38 min (paper contradicts itself) | `data/raw/pubmed_abstracts.jsonl`, `data/raw/qdrant_index/` | 2,784 PubMed abstracts + 18 MedRAG textbook texts + the paper's own Symptom-Disease dataset [30] = **34,206 chunks** (11,244 biomedical + 13,691 textbooks + 9,271 symptom-disease) | ~100× smaller; abstracts+book paragraphs vs full-text PDFs | full-text PDF pipeline (PyMuPDF 65 pg/s, 28 % OCR + ICU normalization, regex section stripping), page metadata | 1,000 PMC PDFs, PyMuPDF, tesseract; **exact >15-keyword queries & ~20 excluded topics NOT published** | ⚠️ tools/data public, exact recipe not |

## B. Section II (intro) — Model lineup

| ID | ① Paper specifies | ② Repo location | ③ Our implementation | ④ Differs | ⑤ Missing | ⑥ Required to reproduce | ⑦ Public? |
|---|---|---|---|---|---|---|---|
| B1 | 4 different LLMs off-the-shelf, **no fine-tuning**: DeepSeek-R1 (Clarifier), Gemini-2.0-Flash (RAG Analyzer + Optimizer), Qwen2-72B (Scanner), GPT-4o-mini (Fusion); embedding `BioBERT-mnli-snli-scinli-scitail-mednli-stsb` | `src/llm.py`, `.env` (`GEMINI_MODEL=`) | **One model `gemini-flash-lite-latest` plays all five roles** (fallback chain: gemini-3.8-flash → gemini-flash-latest → gemini-2.5-flash); embedding model **identical**, 768-D ✓ | documented deviation: 1 model vs 4 — **five prompted roles are not five independent LLMs** | per-role model assignment | 4 model APIs (DeepSeek-R1, Gemini-2.0-Flash, Qwen2-72B, GPT-4o-mini) | ✅ all four public (DeepSeek/Qwen open weights; Gemini/GPT paid APIs) |

## C. Section II-A — Five agents & orchestration

| ID | ① Paper specifies | ② Repo location | ③ Our implementation | ④ Differs | ⑤ Missing | ⑥ Required to reproduce | ⑦ Public? |
|---|---|---|---|---|---|---|---|
| C1 | **Clinical Text Clarifier**: Text Normalization → NER (Symptom/Disease/Medication/Test Result) → Association Extraction → SNOMED CT/ICD-10 mapping → JSON; zero-shot; DeepSeek-R1; cost ≈O(L) | `src/clarifier.py` | Single zero-shot LLM call → JSON schema (`symptoms`, `key_concepts`, `patient_age`, `duration`, …); docstring explicitly delegates entity/relation extraction to the LLM, **no ICD/SNOMED** | no staged NER pipeline; no ontology codes | NER module, association-extraction stage, ontology mapping | NER model (e.g. scispaCy) + ICD-10 / SNOMED | ⚠️ ICD-10 ✅ free; SNOMED CT needs license |
| C2 | **Symptom RAG Analyzer**: cosine top-k ANN search, O(log N)+O(k), *retain only conditions a passage supports*, Gemini-2.0-Flash | `src/rag_analyzer.py`, `src/retrieval.py` | RRF fusion over 2 collections, `TOP_K=5`, evidence-support gate present (`"Drop any condition that no passage supports"` + id check) | RRF multi-collection ranking vs paper's single-index cosine rank; model differs; **paper never states numeric top-k** (5 is our choice) | — | same HF model + Qdrant (both public) | ✅ |
| C3 | **Evidence-Based Scanner**: *proactive continuous* literature scanning; **Serper** → PubMed, Google Scholar, **NICE**; evidence-hierarchy weighting (meta-analyses > case reports); Qwen2-72B | `src/evidence_scanner.py`, `src/pubmed.py` | **NCBI E-utilities** (`esearch`/`efetch`, 0.35 s throttle), query per hypothesis; `evidence_level` map (meta-analysis=1, RCT=2, …) sorted best-first | E-utilities instead of Serper/Scholar/NICE (documented deviation); on-demand per case, not proactive background scanning | Serper, Scholar, NICE integration; background KB refresh | Serper API key (public free tier), NICE pages, Scholar | ⚠️ Serper ✅, NICE ✅, Scholar ❌ no official API |
| C4 | **Clinical Data Fusion**: cross-validation + conflict-detection protocol; calibrated confidence from RAG similarity + web evidence score + cross-source concordance; explainable report; GPT-4o-mini | `src/fusion.py` | LLM agreement/disagreement extraction + **transparent heuristic** `0.35·sim + 0.25·evidence_breadth + 0.20·web_support + 0.20·agreement` (weights are ours) | model differs; calibration formula is our openly-labeled replication heuristic | **paper's actual confidence formula (never published)** | formula unpublished → paper's calibration not exactly reproducible | ❌ |
| C5 | **Adaptive Optimizer**: confidence < τ=0.65 → feedback report → paraphrase/expand query; halt at τ or max 3 iterations; converges in 2–3 cycles, mean 3.7 min on MedQA; Gemini-2.0-Flash | `src/optimizer.py`, `src/pipeline.py`, `src/config.py` | `TAU=0.65` ✓ `MAX_ITERS=3` ✓ structured optimizer prompt ✓ `stop_reason` + `latency_s` logged ✓ | model differs; no cross-case learning | reflexive learning from case similarities | feedback-report content is LLM-generated, unspecified | ❌ |
| C6 | Orchestration: **CrewAI framework** ("Agent, Crew, Process, Task"), sequential process + dynamic feedback loops | `src/pipeline.py` | Explicit Python orchestration (documented deviation) | framework (behavioral equivalence claimed; no CrewAI dependency) | CrewAI integration | CrewAI pip package | ✅ |

## D. Section II-B — Biomedical RAG architecture (Table 2, Figure 5)

| ID | ① Paper specifies | ② Repo location | ③ Our implementation | ④ Differs | ⑤ Missing | ⑥ Required to reproduce | ⑦ Public? |
|---|---|---|---|---|---|---|---|
| D1 | Corpus build: >15-keyword queries + human-study filters, ~20 excluded topics; ≤1,000 PDFs parallel (5.8× speedup, Entrez history); PyMuPDF 65 pg/s, **28 % pages OCR** (ICU normalization), regex stripping (refs/captions/copyright/authors/abstract/keywords); RecursiveCharacterTextSplitter **512/64 with page fallback 500/80**; drop low-content/label-only segments; metadata = filename/page/length | `scripts/fetch_pubmed.py`, `scripts/build_index.py` | 2,784 abstracts; custom separator `split_text` 512/64; **no 500/80 fallback**; **no page metadata** (payload: title/url/pmid/source/topic/pubtypes/year/chunk_index); no OCR stage | abstracts vs PDFs; no page fallback; no OCR; no page/length metadata | full-text PDF pipeline + OCR + fallback + page metadata | PMC OA PDFs, PyMuPDF, tesseract all public; **exact search/topic-exclusion recipe unpublished** | ⚠️ |
| D2 | Embeddings/index: BioBERT-…-stsb, **768-D**, Qdrant, offline one-off indexing, batch upsert, gRPC (−32 % latency); result: 18M embeddings / 200k chunks / 1,000 PDFs, 2.1 GB, 38 min | `src/retrieval.py`, `scripts/build_index.py`, `scripts/build_symptom_index.py` | **Identical HF model** ✓ 768-D COSINE ✓ offline index ✓; in-process path-based Qdrant (no network → gRPC n/a); 34,206 chunks | scale; no server/gRPC mode (cloud URL supported via `.env` for server mode); "18M embeddings from 200k chunks" is arithmetically odd **in the paper** | — | model + Qdrant public; **paper's index not released** | ✅ |
| D3 | §II-B-1: open-label **symptom-disease dataset [30]** (D. Tecblic, HuggingFace) added as third source (labeled, diverse, rare conditions) | not present | index = PubMed abstracts + MedRAG/textbooks only | third source absent | dataset ingestion | HF dataset is public | ✅ but ❌ missing in ours |
| D4 | Figure 5 + §II-B-2: preprocess → **Medical Term Boosting** (Table 2: Active = baseline) → Qdrant search (τ=0.65) → **3-level fallback: Strategy 1 Threshold Reduction → Strategy 2 Simplify Query → Strategy 3 Remove Symptoms** → Results Integration (Score/Source/Info/Document-ID) | `src/retrieval.py::search_with_fallback` (replication Phase 4) | boosted search @τ=0.65 → threshold reduction @0.55 → @0.45 → simplify query @0.45 → remove symptoms @0.45 → no-threshold safety return; hits carry chunk_id/score/rrf/collections/text/title/url/source/topic/pmid/**length**, each rung logs `{strategy, threshold, n_hits, best_score, query_used, ms}` + total `retrieval_ms` | fallback order now matches Fig. 5; term boosting present as a **labeled approximation** (`_boost_query`, algorithm unpublished); page numbers / fragment-length / per-hit timing still absent | exact term-boost weights, per-hit timing | term-boosting algorithm/weights **unpublished**; fallback only described qualitatively | ⚠️ order/strategies replicated, boosting is our interpretation |
| D5 | **Table 2 verbatim**: cosine similarity · 768-D · chunk 512/64 (fallback 500/80) · **term boosting Active (baseline)** · τ=0.65 · max 3 iterations | `src/config.py` | `TAU=0.65` ✓ `MAX_ITERS=3` ✓ chunk 512/64 ✓ cosine ✓ 768-D ✓ (+ `THETA=0.80` from §III ✓) | **2 of 6 Table-2 parameters unimplemented** (term boosting; 500/80 fallback) | term boosting; chunk fallback | both trivially implementable, but boosting spec unpublished | ⚠️ |

## E. Section II-C — Clinical validation & integration

| ID | ① Paper specifies | ② Repo location | ③ Our implementation | ④ Differs | ⑤ Missing | ⑥ Required to reproduce | ⑦ Public? |
|---|---|---|---|---|---|---|---|
| E1 | Validation flow through 3 agents (RAG Analyzer + Evidence Scanner + Data Fusion) with Qdrant tool + **Serper** module; Table 2 params; Fig. 6 demo case | `src/pipeline.py` + 3 modules | Same 3 modules wired end-to-end ✓ | Serper → E-utilities (see C3) | — | — | — |
| E2 | Guarantees: multi-source validation (RAG × web cross-check); contextual confidence score — data below threshold *"selected and not made available to the agent"*; semantic fusion → "best data" | `src/fusion.py`, `src/pipeline.py` | Cross-check ✓ τ-gate ✓; below-threshold cases trigger the **re-analysis loop** (§II-D) rather than being withheld | below-threshold handling (paper says withhold; ours re-queries — closer to §II-D) | — | unpublished details | ❌ |
| E3 | §II-C claim: fused system **94.7 %** vs RAG-only **65.0 %** vs web-only **70.0 %** ("Tables 5–6") | `results/eval_medqa_*.csv` | Our own: **full 90 % / RAG 85 % / baseline 85 %** on MedQA n=20 — honest, labeled, never compared as reproduction | we do not reproduce paper numbers | — (paper-side inconsistency: 65/70 are **Table 7** ablation numbers on 20 q, not Tables 5–6; also 94.7 vs Table 5's 94.0) | — | (flag only) |

## F. Section II-D — Adaptive feedback loop

| ID | ① Paper specifies | ② Repo location | ③ Our implementation | ④ Differs | ⑤ Missing | ⑥ Required to reproduce | ⑦ Public? |
|---|---|---|---|---|---|---|---|
| F1 | τ=0.65 acceptance threshold; feedback = *semantically structured* recommendations (not mechanical repetition); auto-restructure failed queries; alternative queries for ambiguous/missing symptoms; **reflexive learning from case similarities**; max 3 iterations (cost/time reason) | `src/optimizer.py`, `src/pipeline.py` | τ ✓ max 3 ✓ structured optimizer prompt ✓ restructure ✓ `stop_reason=confidence_threshold_met` ✓ | no cross-case learning | case-similarity learning | mechanism unspecified in paper | ❌ |

## G. Section III — Experiments & Results (Tables 3–7)

| ID | ① Paper specifies | ② Repo location | ③ Our implementation | ④ Differs | ⑤ Missing | ⑥ Required to reproduce | ⑦ Public? |
|---|---|---|---|---|---|---|---|
| G1 | Hardware: HP Victus 16, i5-14500HX (14c), 16 GB RAM, RTX 4050 8 GB, Win11, 954 GB SSD; "latest" PyTorch/Transformers/Qdrant; cost ≈O(k+R) | `requirements.txt`, `README.md` | Runs on lab/dev Windows 11 PCs with pinned requirements | hardware differs → **timings not comparable** | — | commodity hardware | ✅ |
| G2 | Prototype: single-page interface + **REST APIs**; annotated report (source, variation sources, confidence) auto-appended; auditable logs; **2–4 min/case**; synthetic de-identified cases; **role-based access** | `web_demo.py` + `web/`, `main.py`, `results/execution_logs/` | Localhost stdlib-HTTP demo + CLI; JSON case reports; execution logs; synthetic MedQA cases; `latency_s` per run | no REST framework; no auth/roles; case time = seconds not minutes (**no web crawl**) | role-based access control | trivial | ✅ |
| G3 | **Table 3**: MedQA 50 (MCQ A–E) · PubMedQA 50 (free-text) · MedBullets 50 (free-text); equal n for fair comparison | `data/samples/*.jsonl` | Frozen 50/50/50 ✓ (MedQA 5 options, PubMedQA free-text 0 options, MedBullets 4–5 options) | our **executed** eval = MedQA n=20 × 3 modes (60 rows, 0 errors: 17/17/18 of 20) | PubMedQA + MedBullets runs (optional, **pending approval — recommend skip**) | datasets public on HuggingFace; runs cost API calls | ✅ datasets |
| G4 | **Metrics**: micro P/R/F1 + specificity from TP/FP/TN/FN; MedQA one-vs-all (correct → TP=1,TN=4; wrong → FP=1,FN=1,TN=3) ⇒ micro-P/R/F1 ≡ accuracy by construction; free-text cosine **θ=0.80** fixed a priori on a held-out dev split | `evaluation/run_eval.py`, `src/config.py` | **Accuracy only** (`correct/n`); θ=0.80 implemented for free-text ✓; grep-verified: no precision/recall/F1/specificity/ablation code | metric set incomplete vs paper | micro P/R/F1/specificity + Table-5-style reporting | formulas **fully published** → small effort; MedQA metrics computable **offline from existing CSVs** (gold+predicted columns, zero API cost) | ✅ formulas public |
| G5 | **Table 4** baselines: 12 rows (Llama3 78.1, GPT-4 79.6, OpenBioLLM-70B 75.1, DeepSeek-R1 92.0, QwQ-32B 78.6, MDAgents 88.7, ER 81.9, Medprompt 82.4, Reconcile 81.3, AutoGen 77.3, DyLAN 64.2, This Study 94.0/88.0/84.0) — *Related Studies* column cites [16],[18],[31]–[34]; agent systems "tested on 50 samples **[31]**" | none | We cite the paper's numbers as **reported-only**; never re-run, never claim reproduction | no baseline re-execution | — | reproducing = re-running **cited studies** (paper itself cites rather than re-runs) — out of scope | ⚠️ models/frameworks public but GPU-heavy; GPT paid |
| G6 | **Table 5**: N=50/50/50; Correct 47/44/42; Acc/P/R/F1 = 0.94/0.88/0.84; Specificity 0.985/0.88/0.84 | none | Not computed (no micro metrics; free-text datasets not run) | — | entire Table-5 computation | needs G4 + free-text runs; exact model versions/predictions unpublished → **not bit-reproducible** | ⚠️ |
| G7 | **Table 6**: 20 MedQA questions × iterations 1–5 → 86.3 %(±1.1)/2.5 min, 91.5 %(±0.8)/3.0, **94.7 %(±0.5)/3.7**, 95.1 %(±0.4)/4.4, 95.5 %(±0.3)/5.5 | `src/pipeline.py` (iterations + `latency_s`) | No iteration sweep executed; `MAX_ITERS=3` (the paper's own cap) | sweep not run | iteration-accuracy/time sweep | protocol partially described — **averaging basis for ±std unstated**; paper runs 4–5 iterations although its cap is 3 | ⚠️ |
| G8 | Timing: RAG ≈1 s + 1 s + 10 s; web ≈45 s/page (1–2 pages → 1–2 min); iteration ≈2–3 min; 2–3 iters ≈3–4 min; 50-q eval ≈2–3 h; model-dependent ≈8–12 s; inter-agent <1 s | `latency_s` in logs | Our runs much faster (no page crawling; E-utilities ≈1–3 s) | timings **not comparable** (no crawler) — documented deviation | web-crawl timing path | Serper/crawler setup (see C3) | ⚠️ |
| G9 | **Table 7 ablation** (20 q): no Clarifier 80.0 · no RAG Analyzer 65.0 · no Scanner 70.0 · no Fusion 60.0 · no Optimizer 70.0 · no Term Boosting 80.0 | **none** (grep-verified: no ablation code) | Partial substitute: 3-mode eval (full / baseline / RAG-only) on n=20 | not an agent-removal ablation | ablation harness (per-agent disable flags + eval modes) | protocol clear (remove one agent, same 20 q) → implementable; results would be ours, not paper's; **term-boosting ablation impossible (feature absent)** | ✅ protocol public |
| G10 | Headline claims: 94/88/84 (Tables 4–5), 86.3→94.7 (Table 6), "+14.2 % output quality", "consistency ×1.8", **"statistically significant"** (§V) | `README.md`, `results/` | We **never** reproduce or assert these; our measured numbers presented separately | reported-only | — | paper reports **no significance test**; +14.2 %/×1.8 derivations unstated | ❌ |

## H. Section IV — Limitations & Discussion

| ID | ① Paper specifies | ② Repo location | ③ Our implementation | ④ Differs | ⑤ Missing | ⑥ Required to reproduce | ⑦ Public? |
|---|---|---|---|---|---|---|---|
| H1 | Limitations: 3 generic datasets ≠ real-world patient coverage; web effectiveness depends on source quality/access; higher compute cost than single models; decision-**support** not diagnostic; hallucination minimized not eliminated. Future: rare/multi-disease cases, official-guidelines integration, source-reliability evaluation, efficiency, multicenter validation, ethics/regulatory | `README.md` (Limitations / Not-yet-evaluated / Future work) | Mirrors all five limitations + adds our honest caveats (live PubMed → non-bit-reproducible; confidence is a heuristic, not calibrated; 5 roles ≠ 5 LLMs) | — | — | — | — |

## I. Section V — Conclusion

| ID | ① Paper specifies | ② Repo location | ③ Our implementation | ④ Differs | ⑤ Missing | ⑥ Required to reproduce | ⑦ Public? |
|---|---|---|---|---|---|---|---|
| I1 | 94 %/88 %/84 %; 86.3→94.7; ablation-confirmed modularity; *"statistically significant performance improvement over state-of-the-art"* | `README.md` | Explicitly labeled **not reproduced**; our 90/85/85 (n=20) presented as our own measurement | reported-only | — | significance test not reported by paper | ❌ |

---

## Paper-internal inconsistencies (flag in defense Q&A — no repo action needed)

1. **Corpus**: intro says 889 articles + 5,630 patient texts; §II-B-1 says 1,000 PDFs → 200k chunks → **18M embeddings** (18M vectors from 200k chunks is arithmetically odd).
2. **94 % vs 94.7 %**: body/abstract say 94 %; Table 6 says 94.7 %; Table 5 = 0.94 (47/50).
3. **§II-C citation wrong**: "94.7 % vs RAG-only 65.0 % and web-only 70.0 % (Tables 5–6)" — the 65/70 values are **Table 7** ablation numbers (20 questions), and Table 6 is the iteration study.
4. **Table 6 granularity**: 86.3 %(±1.1) etc. cannot be integer accuracies of 20 binary questions (5 % granularity); the averaging basis for ±std is never stated.
5. **Iteration cap**: Table 6 runs 4–5 iterations although Table 2 / §II-D set max = 3.
6. **Table 4**: DeepSeek-R1 "no agent" at 92.0 MedQA beats most agent systems; the *Related Studies* column shows baselines are **cited** ([16],[18],[31]–[34]), not re-run — despite the text implying equal-n comparison.
7. **§V claims "statistically significant"** superiority but no statistical test is reported anywhere.
8. Unstated derivations: "+14.2 % output quality", "consistency ×1.8", gRPC "−32 %", "5.8× speedup", "65 pages/s" — measurement setups unspecified.

## Repo nits found while building this table (safe bug/doc fixes)

- ~~`src/retrieval.py:151` docstring says fallback order *"lower threshold -> simplify -> widen k"* but the code does **widen k → simplify**, and neither matches Figure 5~~ — **FIXED in replication Phase 4**: the docstring now describes the implemented Fig. 5 ladder (threshold reduction → simplify → remove symptoms), and widen-k was removed.

---

## Prioritized gap list — for team decision (no work started)

Ranked by (a) reproducibility impact, (b) effort, (c) public availability of inputs.

### Tier 1 — recommended now: high impact, low effort, inputs public, no/low API cost
| # | Gap | Impact | Effort | Cost | Notes |
|---|---|---|---|---|---|
| 1 | **G4**: add micro P/R/F1/specificity to `run_eval.py` (+ Table-5-style summary block) | Lets us report the paper's full metric set honestly | **S** (~1–2 h) | **Free** for MedQA — computable offline from existing CSVs (gold+predicted) | Formulas are published in §III |
| 2 | **G9**: ablation harness — per-agent disable flags + eval modes (no-Clarifier / no-RAG / no-Scanner / no-Fusion / no-Optimizer) | Mirrors Table 7 structure with *our* honest numbers | **M** | 6 variants × 20 q on flash-lite = low API cost (needs go-ahead) | Term-boosting arm impossible (feature absent, see #5) |
| 3 | **Repo nit**: fix `retrieval.py` fallback docstring order | Accuracy of docs | **S** (minutes) | Free | Allowed bug/doc fix |
| 4 | **G7 (mini)**: iteration sweep 1–3 on the same 20 q (paper's own cap) | Table-6 analogue up to the paper's valid range | **M** | 3 passes × 20 q, low API cost | Full 1–5 exceeds the paper's own cap → skip 4–5 or note inconsistency |

### Tier 2 — medium impact/effort, inputs public, needs explicit decision
| # | Gap | Impact | Effort | Cost | Notes |
|---|---|---|---|---|---|
| 5 | **D5/D4**: implement term boosting + Fig. 5 fallback strategies (threshold reduction, remove symptoms) | Closes the only 2 missing Table-2 parameters | **M** | Free (code) | **Algorithm unpublished → implementation would be our interpretation** (borderline scope change) |
| 6 | **G6**: Table-5 metrics for MedQA (after #1) | Paper-style metric reporting | **S** | Free offline | PubMedQA/MedBullets columns need the optional runs (see #12) |

### Tier 3 — high effort / scope change / partial availability — recommend deferring
| # | Gap | Why defer |
|---|---|---|
| 7 | **D1**: full-PDF corpus (1,000 PDFs + OCR + 500/80 fallback + page metadata) | Days of work; exact search recipe unpublished; unlikely to change an n=20 verdict |
| 8 | **C3**: Serper / Google Scholar / NICE web module | Live-web scope change; timing becomes non-reproducible; Scholar has no official API |
| 9 | **C1**: NER pipeline + SNOMED CT/ICD-10 mapping | SNOMED needs a license; high effort; LLM-delegated extraction already works |
| 10 | **G5**: baseline re-runs (Table 4) | GPU-heavy; paper itself cites these numbers rather than re-running them |
| 11 | **B1**: four-model lineup (DeepSeek/Gemini/Qwen/GPT-4o-mini) | 4 API keys + cost + architecture change; deviation already documented |
| 12 | **G3**: PubMedQA + MedBullets evaluations | Optional, require explicit approval — **recommend skipping** (already agreed lean) |

### Decisions needed from the team
1. **Which tier(s) to green-light?** (Recommendation: Tier 1 only — items 1–4.)
2. **PubMedQA/MedBullets runs:** skip? *(recommended: skip)*
3. **Slide deck:** pick one of `docs/SLIDES.md` (7 slides) vs `PROGRESS_PRESENTATION_*` (12 slides) for the defense.
