"""AGENT 4 - Clinical Data Fusion (paper II-A-4 / "Clinical Data Integration").

Compares the RAG stream with the external evidence stream, reports agreement and
conflict, and computes a TRANSPARENT confidence heuristic.

The paper reports tau = 0.65 and says confidence depends on retrieval similarity,
external evidence strength and cross-source agreement, but never publishes the
formula. The heuristic below is therefore ours and is logged component-by-component
so it can be audited (see README, deviations table).
"""
from __future__ import annotations

import json
from typing import Any

from .llm import generate_json

PROMPT = """You are the Clinical Data Fusion agent of a clinical decision support system.
Compare the two evidence streams. Do not invent agreement and do not resolve conflicts silently.

Return ONLY this JSON:
{{
  "agreements": ["..."],
  "disagreements": ["..."],
  "unsupported_claims": ["..."],
  "coverage": [{{"condition": "...", "supported_by": ["rag", "web"]}}],
  "evidence_gaps": ["..."],
  "summary": "3-5 sentence consolidated assessment that cites sources inline as [R#] or [W#]"
}}

CLINICAL QUESTION:
{question}

STRUCTURED PATIENT DATA:
{clinical}

STREAM A - BIOMEDICAL RAG (local BioBERT/Qdrant index):
{rag}

STREAM B - EXTERNAL LITERATURE (PubMed, live search):
{web}
"""


def _clip(text: str, n: int = 700) -> str:
    text = " ".join(str(text or "").split())
    return text[:n]


def _fmt_rag(rag: dict[str, Any]) -> str:
    lines = []
    for h in rag.get("hypotheses", []):
        lines.append(f"- Hypothesis: {h['condition']} | {h['rationale']}")
        for p in h.get("evidence", []):
            lines.append(f"  [R] {p.get('title','')}: {_clip(p.get('text',''))}")
    return "\n".join(lines) or "(no RAG evidence)"


def _fmt_web(web: dict[str, Any]) -> str:
    lines = []
    for r in web.get("records", []):
        lines.append(
            f"- [W] (level {r.get('evidence_level')}, {r.get('year') or 'n.d.'}) "
            f"{r.get('title','')} - {_clip(r.get('abstract',''))}"
        )
    return "\n".join(lines) or "(no external evidence retrieved)"


def _tokens(s: str) -> set[str]:
    return {t for t in "".join(c if c.isalnum() else " " for c in s.lower()).split() if len(t) > 3}


def _web_support(hypotheses: list[dict], records: list[dict]) -> tuple[float, list[str]]:
    if not hypotheses:
        return 0.0, []
    supported = []
    for h in hypotheses:
        cond_tokens = _tokens(h.get("condition", ""))
        if not cond_tokens:
            continue
        for r in records:
            rec_tokens = _tokens(f"{r.get('title','')} {r.get('abstract','')}")
            if len(cond_tokens & rec_tokens) / len(cond_tokens) >= 0.5:
                supported.append(h["condition"])
                break
    return len(supported) / len(hypotheses), supported


def _agreement(rag: dict[str, Any], web_support: list[str]) -> float:
    """Jaccard-style overlap between RAG conditions and web-supported conditions."""
    rag_conds = {h.get("condition", "").lower().strip() for h in rag.get("hypotheses", [])}
    web_conds = {c.lower().strip() for c in web_support}
    rag_conds.discard("")
    if not rag_conds:
        return 0.0
    return len(rag_conds & web_conds) / len(rag_conds)


def fuse(
    question: str,
    clinical: dict[str, Any],
    rag: dict[str, Any],
    web: dict[str, Any],
) -> dict[str, Any]:
    data = generate_json(
        PROMPT.format(
            question=question,
            clinical=json.dumps(clinical, ensure_ascii=False),
            rag=_fmt_rag(rag),
            web=_fmt_web(web),
        )
    )

    hypotheses = rag.get("hypotheses", [])
    records = web.get("records", [])

    sim = float(rag.get("retrieval_similarity") or 0.0)
    sim = max(0.0, min(1.0, sim))
    evidence_breadth = min(1.0, len(rag.get("passages", [])) / max(1, rag.get("top_k", 5)))
    web_support, web_supported_conditions = _web_support(hypotheses, records)
    agreement = _agreement(rag, web_supported_conditions)

    components = {
        "retrieval_similarity": round(sim, 4),
        "evidence_breadth": round(evidence_breadth, 4),
        "web_support": round(web_support, 4),
        "cross_source_agreement": round(agreement, 4),
    }
    confidence = (
        0.35 * sim + 0.25 * evidence_breadth + 0.20 * web_support + 0.20 * agreement
    )

    return {
        "report": data or {},
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
        "confidence_components": components,
        "confidence_method": (
            "0.35*retrieval_similarity + 0.25*evidence_breadth + "
            "0.20*web_support + 0.20*cross_source_agreement (replication heuristic)"
        ),
        "n_rag_passages": len(rag.get("passages", [])),
        "n_web_records": len(records),
        "web_supported_conditions": web_supported_conditions,
    }
