"""AGENT 5 - Adaptive Optimizer (paper II-A-5 / II-D).

Triggers only when confidence < tau and the iteration budget is not exhausted.
Produces a semantically revised query plus a reason, never a blind retry.
"""
from __future__ import annotations

import json
from typing import Any

from .llm import generate_json

PROMPT = """You are the Adaptive Optimizer of a clinical decision support system.
The fused analysis was rejected because its confidence is below the acceptance threshold.
Diagnose WHY the evidence was insufficient, then produce a better retrieval query.

Return ONLY this JSON:
{{
  "reason": "what was missing or conflicting",
  "focus": ["the specific clinical angles to investigate next"],
  "refined_query": "a new 8-15 word biomedical search query, meaningfully different from the previous one"
}}

Rules:
- Do not repeat the previous query with trivial wording changes.
- Expand to differential diagnoses, risk factors, or distinguishing findings when that is what is missing.
- Never invent patient facts.

QUESTION: {question}
THRESHOLD: {tau}
ATTEMPT: {iteration} of {max_iters}
PREVIOUS QUERY: {previous_query}
CONFIDENCE: {confidence}
CONFIDENCE COMPONENTS: {components}
MISSING INFORMATION: {missing}
EVIDENCE GAPS: {gaps}
RAG HYPOTHESES: {hypotheses}
WEB QUERIES TRIED: {web_queries}
"""


def should_optimize(confidence: float, iteration: int, tau: float, max_iters: int) -> bool:
    return confidence < tau and iteration < max_iters


def optimize(
    question: str,
    clinical: dict[str, Any],
    rag: dict[str, Any],
    web: dict[str, Any],
    fusion: dict[str, Any],
    iteration: int,
    tau: float,
    max_iters: int,
) -> dict[str, Any]:
    data = generate_json(
        PROMPT.format(
            question=question,
            tau=tau,
            iteration=iteration,
            max_iters=max_iters,
            previous_query=rag.get("query_used", ""),
            confidence=fusion.get("confidence"),
            components=json.dumps(fusion.get("confidence_components", {})),
            missing=json.dumps(clinical.get("missing_information", []), ensure_ascii=False),
            gaps=json.dumps(
                (fusion.get("report") or {}).get("evidence_gaps", []), ensure_ascii=False
            ),
            hypotheses=json.dumps(
                [h.get("condition") for h in rag.get("hypotheses", [])], ensure_ascii=False
            ),
            web_queries=json.dumps(
                [q.get("query") for q in web.get("queries", [])], ensure_ascii=False
            ),
        )
    )
    out = data or {}
    return {
        "reason": out.get("reason", ""),
        "focus": out.get("focus", []),
        "refined_query": out.get("refined_query") or rag.get("query_used", ""),
    }
