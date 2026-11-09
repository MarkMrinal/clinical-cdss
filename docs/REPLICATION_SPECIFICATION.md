# Replication Specification — what the paper actually specifies

**Source of truth:** Öğdü, Arslanoğlu, Karaköse, *"An Adaptive Multi-Agent LLM-Based Clinical Decision Support System Integrating Biomedical RAG and Web Intelligence"*, IEEE Access vol. 13, 2025, DOI `10.1109/ACCESS.2025.3613340`.
Extracted directly from the original PDF (15 pages) — full text, plus OCR of all 7 tables and Figures 1–6 (the tables have no text layer; vector outlines only).

**Citation format used below:** `§section` = paper section · `p.N` = PDF page (1–15) · `j.NNNNN` = journal page number (printed in footer; mapping: journal = 167389 + PDF page).

**Note on equations:** the paper contains **no numbered equations**. All formulas appear inline (complexity expressions, cosine similarity, the one-vs-all reduction rule). See §M below.

---

## A. Five-agent architecture and responsibilities

**Citation:** §II-A, Fig. 2 (p.4 / j.167393); per-agent subsections §II-A-1…5 (pp.5–6 / j.167394–167395); Fig. 6 (p.10 / j.167399).

- The system consists of **five domain-specialist autonomous agents** in a multi-layered model with *sequential process flows and dynamic feedback loops* (§II-A, p.4).
- **Information flow (Fig. 2, verified by OCR):** `Raw clinical narrative → Clinical Text Clarifier Agent → (Symptom RAG Analyzer Agent ∥ Evidence-Based Scanner Agent) → Clinical Data Fusion Agent → Adaptive Optimizer Agent → Diagnostic results of the patient`.
- Each agent has its own focus area, tasks, abilities and actions.

| # | Agent (paper name) | Responsibility (paper) | Paper's model |
|---|---|---|---|
| 1 | **Clinical Text Clarifier** (§II-A-1) | Normalize + structure unstructured clinical text; highlight critical diagnostic information; make emergency notes legible for downstream agents | DeepSeek-R1 |
| 2 | **Symptom RAG Analyzer** (§II-A-2) | Turn structured symptoms into a query vector; cosine top-k search over the vector DB; propose preliminary diagnoses **retained only if ≥1 retrieved passage supports them** | Gemini-2.0-Flash |
| 3 | **Evidence-Based Scanner** (§II-A-3) | Validate each hypothesis against *current* literature; programmatic searches to academic indexes; categorize gathered information by hierarchy of evidence | Qwen2-72B |
| 4 | **Clinical Data Fusion** (§II-A-4; also called "Clinical Data Integration") | Cross-validate RAG stream vs web stream; conflict-detection protocol; calculate calibrated confidence per hypothesis; produce consolidated, explainable report | GPT-4o-mini |
| 5 | **Adaptive Optimizer** (§II-A-5) | When confidence < τ: identify untrustworthy data, reorganize query parameters, generate a feedback report recommending better queries to agents 2 and 3 | Gemini-2.0-Flash |

## B. Original model assignments and training policy

**Citation:** §II intro (p.4 / j.167393); per-agent paragraphs (pp.5–6).

- **No training or fine-tuning** of any kind: all LLMs used off-the-shelf for inference; updates are made by re-indexing sources, not changing weights.
- Four LLMs total: **DeepSeek-R1, Gemini-2.0-Flash, Qwen2-72B, GPT-4o-mini**.
- Embedding model: **`BioBERT-mnli-snli-scinli-scitail-mednli-stsb`** (Sentence Transformer) used *only* to generate embeddings, weights frozen.
- Model selection rationales stated per agent (schema concordance for Clarifier; long-context alignment for RAG; summarization for Scanner; scalable synthesis for Fusion; low-latency iteration for Optimizer).

## C. Orchestration (CrewAI)

**Citation:** §II-A (p.4 / j.167393).

- Architecture "developed within the **CrewAI** framework", modular agents combined through **sequential process flows and dynamic feedback loops**.
- The *"Agent, Crew, Process, Task"* modules of CrewAI define the tasks, abilities and actions of the agents.

## D. Clinical text processing (Clarifier pipeline)

**Citation:** §II-A (p.4) + §II-A-1 (p.5 / j.167394).

Sequential NLP stages, in order:
1. **Text Normalization** — remove noise: abbreviations (e.g. "HT"), spelling errors, punctuation; convert to standard text format.
2. **Named Entity Recognition (NER)** — identify/label clinical entities in predefined classes: **"Symptom," "Disease," "Medication," "Test Result."**
3. **Association Extraction** — establish semantic context between entities (example: linking *"Numbness"* + *"Left Arm"* → "numbness in the left arm").
4. **Terminology mapping** — map all identified clinical concepts to codes in international medical ontologies: **SNOMED CT or ICD-10**.
5. **Structured output** — convert to semantically enriched structured format (JSON) with a predefined schema, usable by other agents/LLMs.
- Performed **zero-shot**; prioritizes preserving meaning of original content; highlights critical diagnostic information by color-coding.
- Complexity stated: ≈O(L) for single-pass normalization with L = input tokens.
- **Fig. 6 example output fields (OCR):** DEMOGRAPHICS, SYMPTOMS, TRAVEL, LAB, IMAGING, STRUCTUREDDX.

## E. SNOMED CT / ICD-10 mapping

**Citation:** §II-A, final stage of Clarifier description (p.5 / j.167394).

- Explicit requirement: *"the agent first maps all identified clinical concepts to codes in international medical ontologies such as SNOMED CT or ICD-10."*
- No further detail is given (no code system version, no mapping method, no library named).

## F. Biomedical corpus construction

**Citation:** §II-B-1 (pp.7–8 / j.167396–167397), Fig. 3 (p.7), plus intro claim (p.4 / j.167393).

**Pipeline (Fig. 3, OCR):** `Medical Articles and Textbooks + Symptom Disease Dataset → Splitting → Cleaning → Embedding → Qdrant DB Collection`.

1. **Sources:** PubMed articles, medical textbooks, symptom-disease datasets; plus an open-label **Symptom-Disease Dataset from HuggingFace** (ref [30], D. Tecblic) added for semantic diversity (p.7–8).
2. **Query construction:** questions with **>15 keywords** (symptom–diagnosis correlation, clinical finding patterns, multi-organ injury), filters biased to *human clinical studies and diagnostic papers*; **~20 topics excluded** (AI applications, cancer biomarkers, pandemic vaccine effects, …).
3. **Download:** up to **1,000 articles in parallel**, ranked by *"relevance"*; Entrez history enabled to avoid overloading NCBI; PDFs indexed locally; **5.8× speedup** over one CPU core.
4. **Extraction:** **PyMuPDF** page-by-page at ~**65 pages/s** (~9.6× faster than pdfplumber); typo correction; **28% of pages required OCR**; **ICU-based Unicode normalization**; regex/keyword filters remove reference lists, figure/table captions, copyright notices, author info, abstracts, keywords, emails.
5. **Segmentation:** **Recursive Character Text Splitter** with `chunk_size=512`, `chunk_overlap=64`; **page-based fallback** `chunk_size=500`, `chunk_overlap=80` when segment count is inadequate; too-brief/low-content/label-only segments removed.
6. **Metadata:** source file name, **page number**, text length (+ later: source/kind of text, unique ID).
7. **Corpus-size claims (mutually inconsistent in the paper):**
   - Intro (p.4): *"889 publicly available medical articles with 5,630 patient texts"*.
   - §II-B-1 (p.7): **18M embeddings from 200,000 chunks for 1,000 PDFs**; index **2.1 GB**; total processing **38 minutes**; gRPC enabled (**−32% latency** vs REST).

## G. BioBERT embeddings

**Citation:** §II-B-1 (p.7 / j.167396).

- Model: **`BioBERT-mnli-snli-scinli-scitail-mednli-stsb`**; every chunk → **768-dimensional** vector; loaded into **Qdrant** with text + metadata.
- Indexing is **offline, one-off preprocessing**; online inference only queries the precomputed index (real-time responsive, no embedding cost at inference).
- Batch processing at load; small chunk sizes/delimiters when chunks are few to minimize content loss.

## H. Chunking and indexing (Table 2 parameters)

**Citation:** Table 2 (p.9 / j.167398), verbatim via OCR:

| Parameter | Value |
|---|---|
| Similarity criterion | Cosine Similarity |
| Embedding dimension | 768-D embedding |
| chunk_size/overlap | **512/64** (fallback: **500/80**) |
| Term boosting | **Active (baseline)** |
| τ (threshold) | **0.65** |
| Max iterations | **3** |

## I. Qdrant retrieval

**Citation:** §II-A-2 (p.5 / j.167394), §II-B-2 (pp.8–9 / j.167397–167398).

- Cosine-similarity search against the BioBERT/Qdrant index; **top-k passages** passed as context (**the paper never states a numeric k**).
- ANN search: time per query ≈ **O(log N) + O(k)** (N = collection size, k = fetched passages).
- Passages ranked by cosine similarity between query embedding and passage embeddings; only top-k used → each proposed diagnosis traceable to a cited text span.
- "Database" = BioBERT–Qdrant index of textbook, guideline and epidemiology passages — **not** a catalogue of diagnoses.
- Retain rule: candidate conditions kept **only when ≥1 retrieved passage explicitly supports them**.
- Query-side preprocessing: free-text input normalized automatically, medical abbreviations processed, key terms recognized.
- Parametric tool: agents can set similarity threshold, maximum number of results, context-based filtering.
- Decision-trace metadata per query: **source, document ID, page, fragment length, retrieval time** (Fig. 5 "Results Integration": Score / Source / Info / Document ID).

## J. Retrieval fallback (Figure 5, three strategies)

**Citation:** §II-B-2 (p.8–9), Fig. 5 (p.9 / j.167398), Table 2.

**Figure 5 flow (verbatim OCR):** `QueryEntry → Query → Preprocessing → Medical Term Boosting → Qdrant Vector Search (Threshold=0.65) → [was a satisfactory result found?] → NO: Strategy 1: Threshold Reduction → Strategy 2: Simplify Query → Strategy 3: Remove Symptoms → YES → Results Integration (Score / Source / Info / Document ID)`.

- Level 1: **lower the similarity threshold** to attempt larger matches.
- Level 2: **linguistically simplify the query** and re-embed.
- Level 3: **reconstruct the query from clinical semantic content** — convert symptom patterns from the initial query into an alternative semantic cluster.
- Explicitly *"not constructed the same way as traditional top-k rotation methods"*.
- **Medical term boosting:** repeated occurrence of high-priority symptoms in the query so they gain importance in embeddings (example: "shortness of breath", "chest pain" assigned higher scores so prioritized documents are marked). Table 2: **Active (baseline)**; Table 7 ablation arm: disabling it → 80.0%.

## K. External evidence sources (Scanner)

**Citation:** §II-A-3 (p.6 / j.167395); §II-C (p.8).

- Instead of a single patient query, the agent **proactively keeps the knowledge base current by continuously scanning medical literature**.
- **Programmatic searches per hypothesis** submitted via *"a web search API like **Serper**"* to scholarly indexes: **PubMed**, **Google Scholar**, and settled clinical guideline databases like **NICE** (National Institute for Health and Care Excellence).
- Interprets returned web pages; derives text summaries pertinent to each hypothesis.
- Model: Qwen2-72B (query generation + dense-text summarization).

## L. Evidence classification and weighting

**Citation:** §II-A-3 (p.6), §II-C (p.8).

- **Evidence hierarchy weighting:** *"higher-value evidence like systematic reviews and meta-analyses are weighted more than lower-value evidence like case reports"*; elsewhere: *"guidelines/systematic reviews > case reports"*.
- (Paper gives no numeric weights — only the ordering.)
- Source credibility evaluation listed as **future work** (§IV, p.14).

## M. Clinical fusion

**Citation:** §II-A-4 (p.6 / j.167395), §II-C (p.8 / j.167397).

- Inputs: two streams — (1) static RAG information, (2) dynamic web-crawled evidence.
- **Cross-validation and consistency check** (example: check whether a textbook treatment still complies with the latest crawled guideline).
- **Conflict-detection protocol** identifies inconsistencies.
- **Calibrated confidence per diagnostic hypothesis** depending on: **RAG similarity score**, **web information evidence score**, and **degree of concordance between the two streams**. *(No formula or weights are published.)*
- **Semantic fusion:** information combined into a single aggregate ("best data") through a consistency check (§II-C).
- **Contextual confidence score:** recommendations evaluated by the Fusion agent; data below the threshold is *"selected and not made available to the agent"* (§II-C wording).
- Output: structured synthesized report with **evidence, source, and final confidence level for each diagnosis** (explainability).
- Paper's own fusion claim (§II-C): *"On MedQA, the fused system reached 94.7%, outperforming RAG-only (65.0%) and web-only (70.0%) … (Tables 5–6)"* — citation is internally wrong (65/70 are Table 7 ablation values; see §P).

## N. Confidence formula

**Citation:** §II-A-4 (p.6), §II-C (p.8), §II-D (pp.9–10).

- Acceptance threshold **τ = 0.65** (stated three times: §II-A, §II-A-5, §II-D).
- Confidence factors (only these are disclosed): RAG similarity, web evidence strength, cross-source agreement (§II-A-4); "contextual confidence score" (§II-C).
- **The actual formula/weights are never published.** → exact replication impossible from the paper alone.

## O. Adaptive optimization and feedback loop

**Citation:** §II-A-5 (p.6), §II-D (pp.9–10 / j.167398–167399).

1. Optimizer evaluates each analysis with confidence score; **results below τ = 0.65 are prompted for feedback**.
2. Iterative structure issues **new guidance reports** to the agents up to a **maximum of 3 iterations** (stated: *"set to a maximum of three cycles… Any more than three iterations would cost and take longer"*).
3. Feedback is *"not merely mechanical repetition"* — semantically structured recommendations from contextual evaluation.
4. Stopping criteria: **(i) combined confidence ≥ τ, or (ii) maximum of three iterations reached** (§II-A-5).
5. Strategy features (§II-D bullet list): automatically restructure queries that fail initial analysis; generate alternative queries for ambiguous/missing symptoms; **"exhibit reflexive behavior that optimizes itself by learning from case similarities over time"**.
6. Query actions: keyword changes, query expansion/reduction, parameter adaptation; the feedback report is sent to **both** the Symptom RAG Analyzer and the Evidence-Based Scanner (§II-A-5).
7. Stated convergence: *"these loops converge in 2–3 cycles, which had a mean response time of 3.7 minutes on the MedQA benchmark"* (§II-A-5).

## P. Iteration limits and stopping criteria (consolidated)

**Citation:** Table 2 (max iterations **3**), §II-D (max three cycles), §II-A-5 (τ or max-3), Table 6 (p.12 — but reports **5 iterations**, an internal inconsistency).

## Q. Web interface / prototype

**Citation:** §III prototype paragraph (pp.10–11 / j.167399–167400).

- End-to-end prototype: symptom/history input through a **single-page interface**; agent structure executed **via REST APIs**; generates an **annotated report**.
- Per run, automatically appended to the report: **source utilized, potential sources of variation, calibrated confidence score**.
- Cases below the confidence threshold prompt a feedback loop to reanalyze.
- Real-time; typical case analysis **2–4 minutes**; all interactions retained in **auditable logs**.
- Security: **synthetic, de-identified cases only**; no persistent patient data; **role-based access**.
- Fig. 6 (p.10 / j.167399) shows the report anatomy: per-agent panels (Clarifier fields → RAG "BEST ARTICLES / TOP DIAGNOSES" with **numeric scores + High/Mod/Low labels** (0.75 High, 0.57 Mod, 0.39 Low) → Web Search "KEY FEATURES" → Fusion "DECISION / GAP / CONFLICT" → Optimizer "RAG TUNE / WEB SEARCH" recommendations → final diagnostic results).

## R. Datasets (Table 3)

**Citation:** Table 3 + §III (p.11 / j.167400), verbatim OCR:

| Dataset | Short technical specification | Number of tests |
|---|---|---|
| **MedQA** | Postgraduate medical knowledge; **multiple choice (A–E)**; diagnostic reasoning measure | **n=50** |
| **PubMedQA** | Derived from biomedical articles; inference from research question + abstract; **free-text answer** | **n=50** |
| **MedBullets** | Applied/basic medicine and case-based short questions; **free-text response** | **n=50** |

- 50 questions per dataset *"under the same settings"* for fair comparison; the paper **never publishes question IDs or a sampling seed** → exact test sets unreproducible.
- MedQA = 5-choice (A–E) per §III.

## S. Baseline configurations

**Citation:** §II-C heading "…AND BASELINES" (p.8), Table 4 (p.12 / j.167401).

- The paper's "baselines" are **comparison systems from the literature** (Table 4), not an internal plain-LLM arm of their own system. Table 4 columns: Datasets (MedQA/PubMedQA/MedBullets), Category, Model Name, Related Studies.
- **Table 4 verbatim (OCR):**

| Category | Model | MedQA | PubMedQA | MedBullets | Related studies |
|---|---|---|---|---|---|
| No agent | Llama 3 | 78.1 | 66.3 | 68.5 | [16],[31] |
| No agent | GPT-4 | 79.6 | 75.4 | 66.2 | [31],[32] |
| No agent | OpenBioLLM-70B | 75.1 | 79.3 | 58.4 | [33] |
| No agent | DeepSeek-R1 | 92.0 | 76.2 | 79.2 | [34] |
| No agent | QwQ-32B | 78.6 | 77.8 | 54.2 | [34] |
| Multi-agent (Single-model) | MDAgents | 88.7 | 75.0 | 80.8 | [18] |
| Multi-agent (Single-model) | ER | 81.9 | 76.0 | 56.0 | [18] |
| Multi-agent (Single-model) | Medprompt | 82.4 | 51.8 | 71.0 | [18] |
| Multi-agent (Single-model) | Reconcile | 81.3 | 79.7 | 59.5 | [18] |
| Multi-agent (Multi-model) | AutoGen | 77.3 | 60.6 | 55.3 | [18] |
| Multi-agent (Multi-model) | DyLAN | 64.2 | 73.6 | 57.3 | [18] |
| **This** | **This Study** | **94.0** | **88.0** | **84.0** | — |

- Text states MDAgents/Reconcile/AutoGen "were all tested on 50 samples **[31]**" → baselines are **cited**, not re-run by the authors.
- **Web-only and RAG-only** appear as the paper's own configurations: §II-C reports fused 94.7% vs **RAG-only 65.0%** vs **web-only 70.0%**; Table 7 identifies these as ablation arms (no RAG Analyzer = 65.0, no Scanner = 70.0).

## T. Evaluation protocol and metrics

**Citation:** §III (p.11 / j.167400); Table 5 (p.12 / j.167401).

- **Accuracy definitions:** exact-match accuracy on the MCQ label (MedQA); semantic-similarity-based correctness for free-text (PubMedQA, MedBullets).
- **Micro metrics** reported in addition to accuracy: precision, recall (sensitivity), F1-score, specificity — from total TP/FP/TN/FN.
- **MedQA one-vs-all reduction (fully specified):** per question, the correct option = positive class, four remaining = negatives. Correct prediction → **TP=1, TN=4**; incorrect → **FP=1, FN=1, TN=3**. Summed over all questions ⇒ micro-P = micro-R = micro-F1 = accuracy **by construction**; specificity = power to reject the four non-gold options.
- **Free-text rule (fully specified):** cosine similarity `s(ŷ,y) = cos(φ(ŷ), φ(y)) = (u·v)/(‖u‖·‖v‖)`, s ∈ [−1,1]; **θ = 0.80 fixed a priori** (balance precision–recall on a held-out development split: lower θ inflated FP, higher θ cut recall). Under this binary criterion micro-P/R/F1 coincide with accuracy (reported anyway in Table 5).
- Up to **three iterations** used on all three datasets.

**Table 5 verbatim (OCR):**

| | MedQA | PubMedQA | MedBullets |
|---|---|---|---|
| N (evaluated) | 50 | 50 | 50 |
| Correct | 47 | 44 | 42 |
| Incorrect | 3 | 6 | 8 |
| Accuracy | 0.94 | **0.84** | **0.88** |
| Precision (micro) | 0.94 | 0.88 | 0.84 |
| Recall (micro) | 0.94 | 0.88 | 0.84 |
| F1-score (micro) | 0.94 | 0.88 | 0.84 |
| Specificity | 0.985 | 0.88 | 0.84 |

*(OCR row alignment: the Accuracy row's middle/right cells are printed 0.84/0.88 while Correct counts 44/42 imply 0.88/0.84 — a transposition in the paper. Precision/Recall/F1 rows follow the counts. Flagged in §P.)*

- Headline results in text: **94% MedQA, 88% PubMedQA, 84% MedBullets** (§III p.11, §V).

## U. Iteration experiment (Table 6)

**Citation:** Table 6 + §III (p.12 / j.167401), verbatim OCR:

| Iteration | Accuracy | Processing time |
|---|---|---|
| 1 | 86.3% (±1.1) | 2.5 minutes |
| 2 | 91.5% (±0.8) | 3.0 minutes |
| 3 | **94.7% (±0.5)** | 3.7 minutes |
| 4 | 95.1% (±0.4) | 4.4 minutes |
| 5 | 95.5% (±0.3) | 5.5 minutes |

- **20 classically prepared questions from MedQA**, posed with different iteration numbers; exact-match on A–E labels; average results presented.
- Text: accuracy rises 86.3% (first pass) → 94.7% (third); marginal gain beyond third is small; processing time grows; iterations 4–5 show minimal change because the loop begins repeating the same suggestions.
- **Inconsistencies:** (a) 4–5 iterations exceed the paper's own max=3; (b) ±std basis never stated (20 binary questions have 5% granularity — 86.3% is not achievable as a single-run integer accuracy).

## V. Agent ablation experiment (Table 7)

**Citation:** Table 7 + §III (p.13 / j.167402), verbatim OCR:

| Component turned off | Accuracy (20 MedQA questions) |
|---|---|
| Clinical Text Clarifier | 80.0% |
| Symptom RAG Analyzer | 65.0% |
| Evidence-Based Scanner | 70.0% |
| Clinical Data Fusion | 60.0% |
| Adaptive Optimizer | 70.0% |
| Medical Term Boosting | 80.0% |

- Protocol: **one agent at a time removed**; remaining system queried with the **same 20 questions**.
- Text highlights: Fusion removal = biggest drop (60.0, "inconsistent or downright wrong outputs"); Optimizer removal → 70.0; RAG Analyzer removal → 65.0; Scanner removal → "more stale and out-of-scope recommendations"; term-boosting off → 80.0.

## W. Hardware, timing and other experimental facts

**Citation:** §III (pp.10–13 / j.167399–167402).

- Hardware: **HP Victus 16**, Intel Core **i5-14500HX** (2.60 GHz, 14 cores), **16 GB RAM** (5600 MT/s), **NVIDIA RTX 4050 8 GB**, **64-bit Windows 11**, 954 GB SSD; latest PyTorch/Transformers/Qdrant.
- Cost model: single-case total time ≈ **O(k + R)** (k = RAG passages, R = crawled web pages).
- Measured timings: RAG request ≈ **1 s**, vector lookup ≈ **1 s**, agent processing ≈ **10 s**; inter-agent transfer **<1 s**; web agent ≈ **45 s/page**, 1–2 extra pages → ≈1–2 min; single iteration ≈ **2–3 min**; 2–3 iterations ≈ **3–4 min**; 50-question evaluation total ≈ **2–3 hours**; model-dependent evaluation time ≈ **8–12 s**.
- Claimed gains: output quality **+14.2%**; system consistency **×1.8** (derivations never given).

## X. Limitations acknowledged by the paper (§IV, pp.13–14)

- Three generic datasets, none covering real-world patient range (future: rare cases, multi-disease complexity).
- Web effectiveness depends on quality/accessibility of actual sources (future: official clinical guidelines, source-reliability evaluation).
- Multi-agent compute cost higher than single models (future: efficient models/cloud).
- Decision **support** only, not a standalone diagnostic tool; physician retains final decision; hallucination risk minimized, not eliminated.
- Future: multicenter clinical validation, resource-efficient optimization, ethical/regulatory dimensions.

## Y. Published results summary (what we must NOT claim to reproduce)

| Claim | Value | Where |
|---|---|---|
| MedQA / PubMedQA / MedBullets accuracy | **94% / 88% / 84%** | Abstract, §III, §V, Tables 4–5 |
| Correct counts (n=50 each) | 47 / 44 / 42 | Table 5 |
| Micro P/R/F1, specificity | 0.94/0.88/0.84 (MedQA); see Table 5 | Table 5 |
| Iteration study peak (20 q) | 94.7% @ 3 iterations, 3.7 min | Table 6 |
| Ablation (20 q) | see §V table above | Table 7 |
| Fused vs RAG-only vs web-only | 94.7 / 65.0 / 70.0 | §II-C (mis-cited) |
| "Statistically significant" superiority | asserted, **no test reported** | §V |
| Output quality / consistency gains | +14.2% / ×1.8 | §III (derivations absent) |

## Z. Paper-internal inconsistencies (replication must note, not silently resolve)

1. **Corpus:** 889 articles + 5,630 patient texts (p.4) vs 1,000 PDFs → 200k chunks → **18M embeddings** (p.7). 18M vectors from 200k chunks is arithmetically unexplained.
2. **Headline:** 94% (text) vs 94.7% (Table 6) vs Table 5's 0.94 (47/50).
3. **§II-C citation:** "94.7% vs RAG-only 65.0% and web-only 70.0% (Tables 5–6)" — 65/70 are **Table 7** values (20 questions).
4. **Table 5 Accuracy row** appears transposed for PubMedQA/MedBullets (0.84/0.88 vs counts 44/42 ⇒ 0.88/0.84).
5. **Table 6** runs 4–5 iterations although Table 2/§II-D cap = 3; ±std basis unstated.
6. **Table 4** baselines are cited from literature (Related Studies column), yet framed as equal-n comparison.
7. **§V** claims "statistically significant" without any reported statistical test.
8. **Unstated derivations:** +14.2%, ×1.8, gRPC −32%, 5.8× speedup, 65 pages/s.
