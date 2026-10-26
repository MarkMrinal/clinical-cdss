# Related work — positioning the base paper

Purpose: show which cell of the design space the base paper occupies, and what our
replication tests. **Verify each row against the original source before submitting** —
this matrix is a working aid, not a citable summary.

| Work | Multi-agent | Biomedical retrieval | Live web evidence | Cross-source fusion | Confidence-gated re-query | Medical QA benchmarks |
|---|---|---|---|---|---|---|
| **Base paper** — Öğdü et al., IEEE Access 2025 | 5 role-specialised agents (CrewAI) | BioBERT + Qdrant, textbook/guideline corpus | Yes (Serper → PubMed/Scholar/NICE) | Yes, explicit conflict detection + confidence | Yes, τ=0.65, ≤3 cycles | MedQA, PubMedQA, MedBullets |
| **KG4Diagnosis** — Zuo et al., 2025 | Hierarchical (GP + expert agents) | Knowledge-graph construction with BioBERT | No | Knowledge-graph consistency | No | Diagnosis tasks |
| **MedAgents** — Tang et al., 2024 | Role-play collaboration (GP, specialists, moderator) | No — internal LLM knowledge | No | Consensus among agents | No | MedQA-style MCQ |
| **MDAgents** — Kim et al., NeurIPS 2024 | Adaptive team size by question difficulty | No | No | Aggregation of agent outputs | No (difficulty routing instead) | MedQA, MedMCQA, NEJM-QA |
| **Reconcile** — Chen et al., 2023 | Round-table consensus across diverse LLMs | No | No | Yes (voting/consensus) | No | General reasoning, not medical |
| **AutoGen** — Wu et al., 2023 | Generic conversational multi-agent framework | No (bring your own tools) | No (bring your own tools) | No | No | Framework, not a medical system |
| **Med-PaLM 2** — Singhal et al., 2023 | Single model | No | No | n/a | No | MedQA, long-form medical QA |

## What this shows

The base paper's claim is **the intersection of four columns** — retrieval + live web +
explicit fusion + a confidence-triggered loop. Neighbouring work covers at most two of
them. That is why the paper itself says it proposes "no truly new algorithmic paradigm":
the contribution is integration, and integration is exactly what is hardest to reproduce
from a methods section without code.

## What our replication can and cannot test

| Claim in the paper | Testable here? | How |
|---|---|---|
| Multi-source fusion beats single-source | Yes | baseline vs RAG-only vs full on identical questions |
| The feedback loop improves accuracy | Partly | iteration count logged per case; 2 cycles max (time budget) |
| Domain-specific embeddings beat generic retrieval | No (not in scope) | would need a second index with a generic embedder |
| Confidence correlates with correctness | Partly | per-case confidence is logged; too few cases for calibration |
| 94.7% / 88% / 84% accuracy | No | different models, corpus and scoring protocol |

## Gap statement for the report

> The paper reports strong accuracy but provides no prompts, no sampling seed, no
> question IDs, and no confidence formula, and evaluates only 50 questions per dataset.
> We therefore reproduce the architecture rather than the numbers, and report our own
> measurements under a fully logged protocol.
