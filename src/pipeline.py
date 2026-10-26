"""End-to-end orchestration of the five agents (paper Fig. 2 / Fig. 6).

Modes:
  baseline : LLM only, no retrieval
  rag      : biomedical retrieval only, no fusion / feedback
  full     : Clarifier -> RAG -> Scanner -> Fusion -> Optimizer loop (paper configuration)
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

from . import clarifier, config, evidence_scanner, fusion, optimizer, rag_analyzer
from .llm import generate_json
from .retrieval import search_with_fallback
from .rag_analyzer import format_passages

SELECT_PROMPT = """You are the final decision stage of a clinical decision support system.

Return ONLY this JSON:
{{"answer": "<option letter>", "rationale": "one sentence grounded in the evidence"}}

Choose exactly one option letter. Use the evidence when it is relevant; if the evidence
does not address the question, reason from clinical knowledge and say so in the rationale.

QUESTION:
{question}

OPTIONS:
{options}

CONSOLIDATED EVIDENCE / REPORT:
{report}
"""

FREETEXT_PROMPT = """Answer the clinical question in 1-3 sentences.

Return ONLY this JSON: {{"answer_text": "...", "rationale": "one sentence"}}

QUESTION: {question}

EVIDENCE:
{report}
"""


def _select(
    question: str, options: dict[str, str] | None, report: str, mode: str
) -> dict[str, Any]:
    if options:
        opts = "\n".join(f"{k}. {v}" for k, v in options.items())
        data = generate_json(
            SELECT_PROMPT.format(question=question, options=opts, report=report)
        )
        letter = str((data or {}).get("answer", "")).strip().upper()[:1]
        if letter not in options:
            letter = ""
        return {
            "answer": letter,
            "answer_text": options.get(letter, ""),
            "rationale": (data or {}).get("rationale", ""),
            "mode": mode,
        }
    data = generate_json(FREETEXT_PROMPT.format(question=question, report=report))
    return {
        "answer": "",
        "answer_text": str((data or {}).get("answer_text", "")).strip(),
        "rationale": (data or {}).get("rationale", ""),
        "mode": mode,
    }


def run_case(
    question: str,
    options: dict[str, str] | None = None,
    mode: str = "full",
    case_id: str | None = None,
    save_log: bool = True,
) -> dict[str, Any]:
    t0 = time.time()
    log: dict[str, Any] = {
        "case_id": case_id or f"case_{int(t0)}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "options": options,
        "mode": mode,
        "parameters": {
            "tau": config.TAU,
            "max_iters": config.MAX_ITERS,
            "top_k": config.TOP_K,
            "embed_model": config.EMBED_MODEL,
            "llm": config.GEMINI_MODEL,
        },
        "iterations": [],
    }

    if mode == "baseline":
        selected = _select(question, options, "(no retrieval - internal knowledge only)", mode)
        log["selected"] = selected
        log["confidence"] = None
        log["n_iterations"] = 0
        log["latency_s"] = round(time.time() - t0, 2)
        if save_log:
            _write_log(log)
        return log

    if mode == "rag":
        passages, trace = search_with_fallback(question)
        report = format_passages(passages)
        selected = _select(question, options, report, mode)
        log["retrieval"] = {"fallback_trace": trace}
        log["passages"] = passages
        log["selected"] = selected
        log["confidence"] = None
        log["n_iterations"] = 1
        log["latency_s"] = round(time.time() - t0, 2)
        if save_log:
            _write_log(log)
        return log

    # ---- full pipeline -----------------------------------------------------
    clinical = clarifier.clarify(question)
    log["clinical"] = clinical
    fused: dict[str, Any] | None = None

    for it in range(1, config.MAX_ITERS + 1):
        rag = rag_analyzer.analyze(clinical)
        web = evidence_scanner.scan(clinical, rag["hypotheses"])
        fused = fusion.fuse(question, clinical, rag, web)

        log["iterations"].append(
            {
                "iteration": it,
                "query": rag.get("query_used"),
                "n_passages": len(rag.get("passages", [])),
                "passages": rag.get("passages", []),
                "n_hypotheses": len(rag.get("hypotheses", [])),
                "hypotheses": [h.get("condition") for h in rag.get("hypotheses", [])],
                "n_web_records": len(web.get("records", [])),
                "web_records": web.get("records", []),
                "web_queries": [q.get("query") for q in web.get("queries", [])],
                "confidence": fused["confidence"],
                "confidence_components": fused["confidence_components"],
                "stop_reason": None,
            }
        )

        if fused["confidence"] >= config.TAU:
            log["iterations"][-1]["stop_reason"] = "confidence_threshold_met"
            break
        if it >= config.MAX_ITERS:
            log["iterations"][-1]["stop_reason"] = "iteration_limit_reached"
            break

        feedback = optimizer.optimize(
            question, clinical, rag, web, fused, it, config.TAU, config.MAX_ITERS
        )
        log["iterations"][-1]["stop_reason"] = "low_confidence_requery"
        log["iterations"][-1]["feedback"] = feedback
        clinical = {
            **clinical,
            "retrieval_query": feedback["refined_query"],
            "key_concepts": feedback.get("focus") or clinical.get("key_concepts", []),
        }

    report = json.dumps(
        {
            "summary": (fused or {}).get("report", {}).get("summary", ""),
            "agreements": (fused or {}).get("report", {}).get("agreements", []),
            "disagreements": (fused or {}).get("report", {}).get("disagreements", []),
            "unsupported_claims": (fused or {}).get("report", {}).get("unsupported_claims", []),
            "evidence_gaps": (fused or {}).get("report", {}).get("evidence_gaps", []),
            "confidence": (fused or {}).get("confidence"),
        },
        ensure_ascii=False,
        indent=1,
    )

    selected = _select(question, options, report, mode)
    log["selected"] = selected
    log["confidence"] = (fused or {}).get("confidence")
    log["confidence_components"] = (fused or {}).get("confidence_components")
    log["confidence_method"] = (fused or {}).get("confidence_method")
    log["n_iterations"] = len(log["iterations"])
    log["latency_s"] = round(time.time() - t0, 2)
    if save_log:
        _write_log(log)
    return log


def _write_log(log: dict[str, Any]) -> None:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(log["case_id"]))
    path = config.LOG_DIR / f"{safe}.json"
    path.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
