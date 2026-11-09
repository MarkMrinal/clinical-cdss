"""AGENT 5 - Adaptive Optimizer (paper II-A-5 / II-D).

Triggers only when confidence < tau and the iteration budget is not exhausted.
Produces a semantically revised query plus a reason, never a blind retry.

The paper's optimizer "identifies untrustworthy data and intelligently reorganizes
query parameters, including keyword changes, query expansion/reduction, and other
key points to investigate, generating a feedback report" (Sec. II-A-5). Our
`optimize()` returns exactly that report as a dict - `reason` (what was missing or
conflicting), `focus` (the key points to investigate), `query_action` (how the
query was reorganized: keywords / expand / reduce / rewrite) and `refined_query`.
The pipeline feeds it to BOTH downstream agents, as the paper requires: the
refined_query becomes the RAG Analyzer's `retrieval_query` and `focus` replaces the
Clarifier's `key_concepts`, which the Evidence Scanner reads when writing its
per-hypothesis queries.
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
  "query_action": "keywords | expand | reduce | rewrite",
  "refined_query": "a new 8-15 word biomedical search query, meaningfully different from the previous one"
}}

Rules:
- Do not repeat the previous query with trivial wording changes.
- Expand to differential diagnoses, risk factors, or distinguishing findings when that is what is missing.
- query_action says HOW the query was reorganized: "keywords" when the key terms
  themselves had to change, "expand" when concepts were added, "reduce" when the
  query was too broad and had to be narrowed, "rewrite" for a structural rewrite.
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

ACTIONS = ("keywords", "expand", "reduce", "rewrite")


def should_optimize(confidence: float, iteration: int, tau: float, max_iters: int) -> bool:
    """Paper Sec. II-D: keep looping while confidence < tau AND iterations < 3."""
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
        ),
        agent="optimizer",
    )
    out = data or {}
    action = str(out.get("query_action") or "").strip().lower()
    if action not in ACTIONS:  # a model that invents an action is normalised
        action = "rewrite"
    return {
        "reason": out.get("reason", ""),
        "focus": out.get("focus", []),
        "query_action": action,
        "refined_query": out.get("refined_query") or rag.get("query_used", ""),
    }
