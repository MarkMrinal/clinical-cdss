# Slides — 7-slide deck (Tuesday milestone)

> Final deck. Numbers are from `results/eval_medqa_20261026_223318.json` and the saved
> execution logs. Demo commands and expected output are in `docs/DEMO.md`.

---

## Slide 1 — Title

**Replicating an adaptive multi-agent clinical decision support system**

Biomedical RAG + live literature + a confidence-gated feedback loop

- Base paper: Öğdü, Arslanoğlu, Karaköse, *IEEE Access* 13 (2025), DOI 10.1109/ACCESS.2025.3613340
- A 2-person course project · 27 October 2026
- One-line claim: **we rebuilt the architecture and measured it ourselves — we did not reproduce the paper's numbers.**

Speaker note: the paper reports 94.7% / 88% / 84% on MedQA / PubMedQA / MedBullets. We
report our own accuracy on a frozen 20-question MedQA sample, with every prompt and log on disk.

---

## Slide 2 — What the paper claims, and what is testable

The paper combines four things that are usually separate:

1. **Five role-specialised agents** — Clarifier, RAG Analyzer, Evidence Scanner, Fusion, Optimizer
2. **Biomedical RAG** — BioBERT embeddings over textbooks + guidelines + literature in Qdrant
3. **Live web intelligence** — Serper → Google Scholar / PubMed / NICE
4. **Adaptive loop** — if fused confidence < τ = 0.65, rewrite the query and repeat (≤ 3 cycles)

Its result table: MedQA 94.7%, PubMedQA 88%, MedBullets 84% (50 questions each).

**What we can test:** does multi-source fusion + the loop beat (a) the LLM alone and
(b) RAG alone, on identical questions?

**What we cannot test:** equality with 94.7%. Different model, different corpus, no
published prompts, no question IDs, no confidence formula.

Speaker note: the contribution is integration, not a new algorithm — which is exactly why
it has to be rebuilt to be checked.

---

## Slide 3 — Architecture as built

```
                       clinical question (MCQ or free text)
                                    │
                     ┌──────────────▼──────────────┐
                     │ 1. Clinical Text Clarifier  │  symptoms, duration, concepts,
                     │    (Gemini, JSON mode)      │  missing info
                     └──────────────┬──────────────┘
                                    │
              ┌─────────────────────▼─────────────────────┐
              │ 2. RAG Analyzer                           │
              │    BioBERT 768-d → Qdrant (cosine)        │
              │    ┌────────────┐   ┌──────────────┐      │
              │    │ biomedical │   │  textbooks   │      │
              │    │ 11,244     │   │  13,691      │      │
              │    └─────┬──────┘   └──────┬───────┘      │
              │          └──── RRF ───────┘              │
              └─────────────────────┬─────────────────────┘
                                    │ top-5 cited passages
              ┌─────────────────────▼─────────────────────┐
              │ 3. Evidence Scanner                       │
              │    PubMed E-utilities (free)              │
              │    evidence hierarchy + PMID/URL          │
              └─────────────────────┬─────────────────────┘
                                    │
              ┌─────────────────────▼─────────────────────┐
              │ 4. Fusion                                 │
              │    agreements / conflicts / gaps          │
              │    confidence = 0.35·sim + 0.25·breadth   │
              │               + 0.20·web + 0.20·agreement │
              └─────────────────────┬─────────────────────┘
                                    │
                         confidence ≥ 0.65 ?
                          ┌─────────┴─────────┐
                         yes                 no
                          │                   │
                          │        ┌──────────▼──────────┐
                          │        │ 5. Optimizer        │
                          │        │ refine query, re-run│
                          │        │ (max 2 in our eval) │
                          │        └──────────┬──────────┘
                          │                   └── back to 2
                          ▼
                 final option + rationale
                 + full JSON audit log
```

Implementation: Python 3.11, `sentence-transformers` (BioBERT), Qdrant embedded mode,
Gemini REST API with disk cache. No CrewAI — the loop is explicit and inspectable.

Speaker note: every passage keeps its PMID or textbook title, so any answer can be traced back.

---

## Slide 4 — Where we deviate, and why

| Paper | This project | Why |
|---|---|---|
| 5 different LLMs (DeepSeek-R1, Gemini-2.0-Flash, Qwen2-72B, GPT-4o-mini) | one Gemini model, five prompts | one API key, course budget |
| Gemini-2.0-Flash | `gemini-flash-lite-latest` | 2.0/2.5 retired on current keys; 3.8-flash returned 503 |
| 1,000 PDFs / 200k chunks / 18M embeddings | 2,784 PubMed abstracts + 13,691 textbook passages = **24,935 chunks** | free, citable; paper's counts are internally inconsistent |
| Serper + Scholar + NICE | PubMed E-utilities | no paid search API; PubMed is the paper's own primary source |
| CrewAI | explicit Python pipeline | debuggable, no hidden agent turns |
| SNOMED CT / ICD-10 | LLM extraction only | licence |
| confidence formula (not published) | documented heuristic | paper gives only τ = 0.65 |
| ≤ 3 feedback cycles | 2 cycles in the timed eval | API at 22–33 s/call |

**These are deviations, not failures.** Each one is logged in the repo and repeated on the
results slide.

---

## Slide 5 — What actually ran

| Check | Result |
|---|---|
| Knowledge base | 24,935 chunks, 768-d, cosine; two collections |
| Retrieval | dual-collection RRF; textbook share of top-5 = **40%** |
| Live evidence | PubMed scanner returns **4 records per condition** (after fixing over-long AND queries) |
| Single MedQA case (`medqa_347`) | **correct** (A, primary hyperparathyroidism), confidence **0.8577**, 1 iteration, **100.6 s** |
| Adaptive MedQA case (`medqa_107`) | first pass: 0 hypotheses, 0 PubMed records, confidence **0.4499** → optimizer rewrote the query → second pass: Klinefelter hypothesis, 4 records, confidence **0.8277**, **correct** (A), 2 iterations |
| Synthetic pneumonia case | first hypothesis wrong → optimizer requery → **correct**, 2 iterations, **268 s**; confidence ended at **0.4794**, below τ, so it stopped at the iteration cap |
| Auditability | every case writes a JSON log with passages, PMIDs, queries, confidence components |

The `medqa_107` log is the strongest demonstration: the first pass produced no usable
hypothesis or web evidence at all; the Optimizer called the query a "keyword dump" and
rewrote it around Klinefelter syndrome; the second pass found four PubMed records and
answered correctly. That is the paper's central mechanism doing real work.

Speaker note: single-case success is a smoke test, not evidence of accuracy — slide 6 is the evidence.

---

## Slide 6 — Results (20 frozen MedQA questions × 3 configurations)

| Configuration | Correct | Accuracy | Errors | Mean confidence | Mean iterations | Latency / case* |
|---|---|---|---|---|---|---|
| Baseline (LLM only) | 17/20 | 85.0% | 0 | — | 0 | 28.8 s |
| RAG only | 17/20 | 85.0% | 0 | — | 1 | ~32 s |
| Full (RAG + web + fusion + loop) | **18/20** | **90.0%** | 0 | 0.791 | 1.5 | 144.1 s |

\* Cold-cache first run. The clean rerun used cached LLM responses (13 s/case for full) and
is **not** deployment latency. Accuracy, errors and confidence are from the clean rerun
(`results/eval_medqa_20261026_223318.json`), 4 workers, same 20 questions.

- Same 20 questions in all rows, seed 42, exact match on the option letter.
- **Full fixed one case and broke none:** `medqa_107` was wrong at baseline (B) and in
  RAG-only (D), and correct in full (A) after 2 iterations.
- **RAG alone exactly matched baseline:** same 17/20, same three errors (`107`, `202`, `376`).
  On this sample, retrieval by itself did not change any decision.
- The two full-mode errors were high-confidence: `202` at 0.855 and `376` at 0.849.
  Confidence measures cross-source agreement, not correctness.
- **+1/20 is not statistically significant.** This is a directional result, not evidence
  that the pipeline is generally better.
- The first RAG run had 3 Qdrant concurrency failures (14/20 valid, 70% on those runs);
  after fixing the client-init race and rerunning, RAG is clean at 17/20 with 0 errors.

Speaker note: the paper reports 94.7%. Our 90% is 20 questions, a lite model, a 25k-chunk
corpus and a different scoring protocol. Do not present it as a reproduction.

---

## Slide 7 — What we learned, and what is next

**Findings**

1. **Retrieval did not add accuracy on this sample.** RAG-only matched the model alone
   exactly (17/20, same three errors). A lexical check finds the gold answer string in the
   top-5 for 0/20 MedQA items; adding 13,691 textbook passages did not change that. Exam
   items often paraphrase the answer, so the check understates retrieval — but the RAG-only
   row is the stronger evidence that retrieval was not the source of the one gain.
2. **Infrastructure dominates the schedule.** Model retirement (2.0 → 3.8 → lite), 22–33 s
   per call, and a CPU-only BioBERT encoder set the pace; caching made reruns free.
3. **The loop demonstrably fires.** `medqa_107` is the clean example: first pass produced
   zero hypotheses and zero PubMed records at confidence 0.4499; the Optimizer rewrote the
   query; the second pass found Klinefelter evidence and answered correctly at 0.8277.
   `medqa_347` stopped after one iteration at 0.8577. On the synthetic pneumonia case the
   correction came from the re-query, but confidence stayed below τ and hit the loop cap.

**Next**

- 50-question runs and a second dataset (PubMedQA) with the θ = 0.80 rule labelled as lenient
- A better retrieval metric (answer-containing chunk with stemming / embedding match)
- Two-model heterogeneity once a second API key is available

**Demo A (normal):** `python main.py --case medqa_347`
**Demo B (adaptive correction):** `python main.py --case medqa_107`

---

## Backup slide (not presented) — paper inconsistencies we found

- "RAG-only 65.0% / web-only 70.0%" are Table 7 **ablation** rows (remove RAG agent,
  remove Optimizer), not a RAG-vs-web comparison.
- Table 6's 86.3% → 94.7% over iterations cannot hold on 20 exact-match questions.
- MedBullets / PubMedQA are scored with a free-text cosine rule while MedQA uses exact
  match — the three accuracies are not on one scale.
- Corpus is described as 889 documents, 5,630 chunks, 1,000 PDFs, and 200k chunks / 18M
  embeddings in different places; these cannot all be true.
