"""End-to-end orchestration of the five agents (paper Fig. 2 / Fig. 6).

Modes:
  baseline : LLM only, no retrieval
  rag      : biomedical retrieval only, no fusion / feedback
  web      : external evidence only (PubMed + NICE + Serper), no local index and
             no fusion loop - the paper's web-only / "no RAG Analyzer" arm
  full     : Clarifier -> RAG -> Scanner -> Fusion -> Optimizer loop (paper configuration)

The two ablation arms (rag, web) are single-pass by design: Fusion exists to
cross-check TWO streams, so with one stream there is nothing to fuse - symmetric
with the paper's own ablation structure and with each other.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

from . import clarifier, config, evidence_scanner, fusion, optimizer, rag_analyzer
from .llm import generate_json, reset_trace, take_trace
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
            SELECT_PROMPT.format(question=question, options=opts, report=report),
            agent="select",
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
    data = generate_json(FREETEXT_PROMPT.format(question=question, report=report), agent="select")
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
    if mode not in ("baseline", "rag", "web", "full"):
        # without this guard an unknown --mode would silently fall through to the
        # full pipeline and pollute an evaluation run
        raise ValueError(f"unknown mode {mode!r}; expected baseline|rag|web|full")
    reset_trace()  # per-run model-execution trace (thread-local)
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
            "agent_config": config.AGENT_CONFIG,
        },
        "iterations": [],
    }

    if mode == "baseline":
        selected = _select(question, options, "(no retrieval - internal knowledge only)", mode)
        log["selected"] = selected
        log["confidence"] = None
        log["n_iterations"] = 0
        log["model_trace"] = take_trace()
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
        log["model_trace"] = take_trace()
        log["latency_s"] = round(time.time() - t0, 2)
        if save_log:
            _write_log(log)
        return log

    if mode == "web":
        # Paper Sec. II-C web-only arm: external evidence only. No local index is
        # touched (no Qdrant search at all) and, with a single stream, no fusion
        # pass - symmetric with the "rag" arm above.
        clinical = clarifier.clarify(question)
        log["clinical"] = clinical
        hypotheses = rag_analyzer.propose_without_retrieval(clinical)
        web = evidence_scanner.scan(clinical, hypotheses)

        candidate_lines = "\n".join(
            f"- {h['condition']}"
            + (f" | {h['rationale']}" if h.get("rationale") else "")
            for h in hypotheses
        ) or "(no candidate conditions proposed)"
        report = (
            "EXTERNAL EVIDENCE ONLY - web-only configuration "
            "(no local RAG index, no fusion loop):\n\n"
            "CANDIDATE CONDITIONS (proposed from the clinical input, no retrieval):\n"
            f"{candidate_lines}\n\n"
            f"EXTERNAL EVIDENCE (PubMed, NICE guidelines, web search):\n"
            f"{fusion.format_web(web)}"
        )

        log["iterations"].append(
            {
                "iteration": 1,
                "query": None,
                "retrieval_trace": None,  # proves no local index was consulted
                "n_passages": 0,
                "passages": [],
                "n_hypotheses": len(hypotheses),
                "hypotheses": [h.get("condition") for h in hypotheses],
                "n_web_records": len(web.get("records", [])),
                "web_records": web.get("records", []),
                "web_sources": web.get("sources", {}),
                "web_summaries": web.get("web_summaries", []),
                "search_failures": web.get("search_failures", []),
                "web_queries": [q.get("query") for q in web.get("queries", [])],
                "confidence": None,
                "confidence_components": None,
                "stop_reason": "single_pass_web_only",
            }
        )
        log["report"] = {
            "stream": "web_only",
            "hypotheses": [h.get("condition") for h in hypotheses],
            "n_web_records": len(web.get("records", [])),
            "web_sources": web.get("sources", {}),
            "web_summaries": web.get("web_summaries", []),
            "confidence": None,
        }
        selected = _select(question, options, report, mode)
        log["selected"] = selected
        log["confidence"] = None
        log["n_iterations"] = 1
        log["model_trace"] = take_trace()
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
                # Fig. 5 chain actually used for this iteration (strategy ladder,
                # thresholds, per-step ms) - needed to audit the fallback behaviour
                "retrieval_trace": rag.get("fallback_trace"),
                "n_passages": len(rag.get("passages", [])),
                "passages": rag.get("passages", []),
                "n_hypotheses": len(rag.get("hypotheses", [])),
                "hypotheses": [h.get("condition") for h in rag.get("hypotheses", [])],
                "n_web_records": len(web.get("records", [])),
                "web_records": web.get("records", []),
                # Phase 5 audit fields: which sources answered, why any failed,
                # and the derived per-hypothesis web summaries
                "web_sources": web.get("sources", {}),
                "web_summaries": web.get("web_summaries", []),
                "search_failures": web.get("search_failures", []),
                "web_queries": [q.get("query") for q in web.get("queries", [])],
                "confidence": fused["confidence"],
                "confidence_components": fused["confidence_components"],
                # Phase 6 audit fields: the paper's three named factors and the
                # per-diagnosis evidence/source/confidence rows (Sec. II-A-4)
                "confidence_factors": fused.get("confidence_factors"),
                "diagnoses": fused.get("diagnoses"),
                "stop_reason": None,
            }
        )

        # Paper Sec. II-D: the loop runs while confidence < tau and fewer than
        # three iterations have been used (single source of truth: optimizer.py)
        if not optimizer.should_optimize(
            fused["confidence"], it, config.TAU, config.MAX_ITERS
        ):
            log["iterations"][-1]["stop_reason"] = (
                "confidence_threshold_met"
                if fused["confidence"] >= config.TAU
                else "iteration_limit_reached"
            )
            break

        # The optimizer's feedback report reaches BOTH downstream agents: the
        # refined_query becomes the RAG Analyzer's retrieval_query and `focus`
        # replaces key_concepts, which the Evidence Scanner reads (paper II-A-5).
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

    # The consolidated report. Per paper Sec. II-A-4 it must summarise "the
    # evidence, source, and final confidence level for each diagnosis", so the
    # deterministic per-diagnosis rows ride along with the model's narrative.
    # This dict is both what the selection stage reads and what the log keeps
    # (log["report"]), so the two can never drift apart.
    report_obj = {
        "summary": (fused or {}).get("report", {}).get("summary", ""),
        "agreements": (fused or {}).get("report", {}).get("agreements", []),
        "disagreements": (fused or {}).get("report", {}).get("disagreements", []),
        "unsupported_claims": (fused or {}).get("report", {}).get("unsupported_claims", []),
        "evidence_gaps": (fused or {}).get("report", {}).get("evidence_gaps", []),
        "diagnoses": (fused or {}).get("diagnoses", []),
        "confidence": (fused or {}).get("confidence"),
    }
    log["report"] = report_obj

    selected = _select(
        question,
        options,
        json.dumps(report_obj, ensure_ascii=False, indent=1),
        mode,
    )
    log["selected"] = selected
    log["confidence"] = (fused or {}).get("confidence")
    log["confidence_components"] = (fused or {}).get("confidence_components")
    log["confidence_factors"] = (fused or {}).get("confidence_factors")
    log["diagnoses"] = (fused or {}).get("diagnoses")
    log["confidence_method"] = (fused or {}).get("confidence_method")
    log["n_iterations"] = len(log["iterations"])
    log["model_trace"] = take_trace()
    log["latency_s"] = round(time.time() - t0, 2)
    if save_log:
        _write_log(log)
    return log


def _write_log(log: dict[str, Any]) -> None:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(log["case_id"]))
    path = config.LOG_DIR / f"{safe}.json"
    path.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
