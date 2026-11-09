"""AGENT 4 - Clinical Data Fusion (paper II-A-4 / "Clinical Data Integration").

Compares the RAG stream with the external evidence stream, reports agreement and
conflict, and computes a TRANSPARENT confidence heuristic.

The paper reports tau = 0.65 and says the calibrated confidence "depends on factors
such as the similarity score of the RAG, evidence score of web information, and
degree of concordance between the two streams" (Sec. II-A-4) - but never publishes
the formula. Replication Phase 6 therefore maps our four components ONTO exactly
those three named factors (`confidence_factors`) while keeping the arithmetic
byte-identical to the pre-Phase-6 heuristic, and adds the per-diagnosis rows the
paper asks for in the report ("the evidence, source, and final confidence level for
each diagnosis"). Weights stay ours and are labelled as such everywhere.
"""
from __future__ import annotations

import json
from typing import Any

from . import config
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

STREAM B - EXTERNAL EVIDENCE (PubMed, NICE guidelines, web search - live):
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


def format_web(web: dict[str, Any]) -> str:
    """Render stream B as `[W1]..[Wn]` lines (+ derived summaries).

    Public because the web-only ablation arm (pipeline mode="web") shows exactly
    this external stream on its own, with no fusion pass.
    """
    lines = []
    for i, r in enumerate(web.get("records", []), 1):
        src = r.get("source") or "pubmed"
        lines.append(
            f"- [W{i}] ({src}, level {r.get('evidence_level')}, {r.get('year') or 'n.d.'}) "
            f"{r.get('title','')} - {_clip(r.get('abstract',''))}"
        )
    for s in web.get("web_summaries", []) or []:
        lines.append(f"- [summary] {s.get('condition','')}: {s.get('summary','')}")
    return "\n".join(lines) or "(no external evidence retrieved)"


def _tokens(s: str) -> set[str]:
    return {t for t in "".join(c if c.isalnum() else " " for c in s.lower()).split() if len(t) > 3}


def _web_support(
    hypotheses: list[dict], records: list[dict]
) -> tuple[float, list[str], dict[str, list[int]]]:
    """Token-overlap support test (>= 0.5 of the condition's content tokens).

    Returns (ratio, supported_conditions, {condition: [1-based record indices]}).
    The indices are the numbering used by format_web() ([W1]..[Wn]), so a report
    row can cite exactly which web records back which diagnosis. The supported
    set - and therefore every published number - is unchanged from Phase 5.
    """
    if not hypotheses:
        return 0.0, [], {}
    supported = []
    matches: dict[str, list[int]] = {}
    for h in hypotheses:
        cond = h.get("condition", "")
        cond_tokens = _tokens(cond)
        if not cond_tokens:
            continue
        hits = []
        for i, r in enumerate(records, 1):
            rec_tokens = _tokens(f"{r.get('title','')} {r.get('abstract','')}")
            overlap = len(cond_tokens & rec_tokens) / len(cond_tokens)
            if overlap >= 0.5:
                hits.append(i)
        if hits:
            supported.append(cond)
            matches[cond] = hits
    return len(supported) / len(hypotheses), supported, matches


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
            web=format_web(web),
        ),
        agent="fusion",
    )

    hypotheses = rag.get("hypotheses", [])
    records = web.get("records", [])

    sim = float(rag.get("retrieval_similarity") or 0.0)
    sim = max(0.0, min(1.0, sim))
    evidence_breadth = min(1.0, len(rag.get("passages", [])) / max(1, rag.get("top_k", 5)))
    web_support, web_supported_conditions, web_matches = _web_support(hypotheses, records)
    agreement = _agreement(rag, web_supported_conditions)

    components = {
        "retrieval_similarity": round(sim, 4),
        "evidence_breadth": round(evidence_breadth, 4),
        "web_support": round(web_support, 4),
        "cross_source_agreement": round(agreement, 4),
    }

    # --- paper Sec. II-A-4: the three NAMED factors -------------------------
    # "It depends on factors such as the similarity score of the RAG, evidence
    # score of web information, and degree of concordance between the two streams."
    # Our four components are grouped onto exactly those three factors. The
    # grouping is purely presentational: the sum below is the same left-to-right
    # expression as the pre-Phase-6 heuristic, so no published number moves and
    # the optimizer's trigger decisions are unchanged. The 0.35/0.25/0.20/0.20
    # weights are OURS (the paper never publishes its formula).
    rag_factor = 0.35 * sim + 0.25 * evidence_breadth
    web_factor = 0.20 * web_support
    concordance_factor = 0.20 * agreement
    confidence = rag_factor + web_factor + concordance_factor

    factors = {
        "rag_similarity": {
            "paper_factor": "similarity score of the RAG (paper Sec. II-A-4)",
            "components": {"retrieval_similarity": round(sim, 4),
                           "evidence_breadth": round(evidence_breadth, 4)},
            "weights": {"retrieval_similarity": 0.35, "evidence_breadth": 0.25},
            "contribution": round(rag_factor, 4),
        },
        "web_evidence_score": {
            "paper_factor": "evidence score of web information (paper Sec. II-A-4)",
            "components": {"web_support": round(web_support, 4)},
            "weights": {"web_support": 0.20},
            "contribution": round(web_factor, 4),
        },
        "cross_stream_concordance": {
            "paper_factor": (
                "degree of concordance between the two streams (paper Sec. II-A-4)"
            ),
            "components": {"cross_source_agreement": round(agreement, 4)},
            "weights": {"cross_source_agreement": 0.20},
            "contribution": round(concordance_factor, 4),
        },
    }

    # --- per-diagnosis report rows (paper Sec. II-A-4) ----------------------
    # "... a structured, synthesized report summarizing the evidence, source, and
    # final confidence level for each diagnosis adds explainability". Deterministic
    # (no extra LLM call): each row reuses the SAME weights as the aggregate, with
    # the web terms evaluated for that one diagnosis, so the rows average out to
    # the aggregate confidence whenever the condition names are unique.
    rag_base = rag_factor
    diagnoses: list[dict[str, Any]] = []
    for h in hypotheses:
        cond = str(h.get("condition", ""))
        web_flag = 1.0 if cond in web_supported_conditions else 0.0
        # a RAG hypothesis corroborated by the web stream = both streams concur
        concordance_flag = web_flag
        conf_h = rag_base + 0.20 * web_flag + 0.20 * concordance_flag
        evidence = [f"R{i}" for i in h.get("evidence_ids", []) if isinstance(i, int)]
        evidence += [f"W{i}" for i in web_matches.get(cond, [])]
        diagnoses.append(
            {
                "condition": cond,
                "rationale": h.get("rationale", ""),
                "source": ["rag"] + (["web"] if web_flag else []),
                "evidence": evidence,  # [R#] = local passage, [W#] = web record
                "confidence": round(conf_h, 4),
                # paper Sec. II-C: recommendations below tau are flagged, not
                # silently served (they drive the Sec. II-D feedback loop instead)
                "above_threshold": conf_h >= config.TAU,
            }
        )

    return {
        "report": data or {},
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
        "confidence_components": components,
        "confidence_factors": factors,
        "confidence_method": (
            "0.35*retrieval_similarity + 0.25*evidence_breadth + "
            "0.20*web_support + 0.20*cross_source_agreement - our heuristic, "
            "grouped onto the paper's three named factors (Sec. II-A-4: "
            "rag_similarity / web_evidence_score / cross_stream_concordance); "
            "the paper never publishes its formula, so the weights are ours"
        ),
        "diagnoses": diagnoses,
        "n_rag_passages": len(rag.get("passages", [])),
        "n_web_records": len(records),
        "web_supported_conditions": web_supported_conditions,
    }
