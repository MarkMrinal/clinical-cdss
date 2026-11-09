"""AGENT 2 - Symptom RAG Analyzer (paper II-A-2).

Retrieve passages from the BioBERT/Qdrant index, then propose hypotheses that are
explicitly tied to retrieved evidence. A hypothesis without a supporting passage
is dropped, per the paper's rule.
"""
from __future__ import annotations

import json
from typing import Any

from . import config
from .llm import generate_json
from .retrieval import search_with_fallback

PROMPT = """You are the Symptom RAG Analyzer of a clinical decision support system.
You may ONLY use the retrieved passages below as evidence.

Return ONLY this JSON:
{{
  "hypotheses": [
    {{"condition": "name", "rationale": "one sentence", "evidence_ids": [1, 2]}}
  ],
  "uncertainties": ["what the retrieved evidence does not settle"]
}}

Rules:
- Every hypothesis MUST cite at least one passage number in evidence_ids.
- Drop any condition that no passage supports.
- Do not use outside knowledge. If the passages are insufficient, return fewer hypotheses
  and say so in uncertainties.
- Maximum 3 hypotheses, ordered by likelihood.

CLINICAL INPUT:
{clinical}

RETRIEVED PASSAGES:
{passages}
"""


def format_passages(passages: list[dict]) -> str:
    lines = []
    for i, p in enumerate(passages, 1):
        lines.append(
            f"[{i}] {p.get('title','')} ({p.get('journal') or p.get('source','')}, {p.get('year','')})\n{p.get('text','')}"
        )
    return "\n\n".join(lines)


def analyze(clinical: dict[str, Any]) -> dict[str, Any]:
    query = clinical.get("retrieval_query") or " ".join(clinical.get("symptoms") or [])
    symptoms = [str(s) for s in clinical.get("symptoms") or [] if str(s).strip()]
    concepts = [str(c) for c in clinical.get("key_concepts") or [] if str(c).strip()]
    # Paper Table 2: term boosting is Active on the user-supplied terms.
    # _boost_query() drops negated terms ("no ...") and caps at 5 terms / 400 chars.
    boost_terms = concepts + symptoms
    # Strategy 2 (Fig. 5): a concept-level query with the symptom prose removed.
    simplified = " ".join((concepts or symptoms)[:6])
    passages, trace = search_with_fallback(
        query,
        simplified_query=simplified or None,
        boost_terms=boost_terms,
        symptom_terms=symptoms,
    )

    data = generate_json(
        PROMPT.format(
            clinical=json.dumps(clinical, ensure_ascii=False),
            passages=format_passages(passages),
        ),
        agent="rag_analyzer",
    )

    hypotheses = []
    for h in (data or {}).get("hypotheses", []):
        ids = [i for i in h.get("evidence_ids", []) if isinstance(i, int) and 1 <= i <= len(passages)]
        if not ids:  # paper: retain only evidence-backed conditions
            continue
        hypotheses.append(
            {
                "condition": h.get("condition", ""),
                "rationale": h.get("rationale", ""),
                "evidence_ids": ids,
                "evidence": [passages[i - 1] for i in ids],
            }
        )

    return {
        "query_used": query,
        "fallback_trace": trace,
        "passages": passages,
        "hypotheses": hypotheses,
        "uncertainties": (data or {}).get("uncertainties", []),
        "retrieval_similarity": round(
            sum(p["score"] for p in passages) / len(passages), 4
        )
        if passages
        else 0.0,
        "top_k": config.TOP_K,
    }
