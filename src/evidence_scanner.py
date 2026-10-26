"""AGENT 3 - Evidence-Based Scanner (paper II-A-3).

Queries PubMed for current literature on each hypothesis, keeps titles/abstracts/
identifiers, and tags an evidence level so stronger designs outrank case reports.
Substitute for the paper's Serper + Google Scholar + NICE access (documented deviation).
"""
from __future__ import annotations

import json
from typing import Any

from . import config, pubmed
from .llm import generate_json

QUERY_PROMPT = """You are the Evidence-Based Scanner of a clinical decision support system.
Write ONE PubMed search query (max 12 words, no boolean operators except AND) that would find
the strongest current evidence for this candidate condition, given the patient context.

Return ONLY JSON: {{"query": "..."}}

Candidate condition: {condition}
Patient context: {clinical}
"""


def _build_query(condition: str, clinical: dict[str, Any]) -> str:
    if not config.SCANNER_LLM_QUERY:
        # PubMed ANDs terms, so long keyword strings return nothing. Use the condition
        # alone (relevance-sorted). Documented simplification of the paper's Qwen query writer.
        return condition.strip()
    try:
        data = generate_json(
            QUERY_PROMPT.format(
                condition=condition,
                clinical=json.dumps(
                    {
                        k: clinical.get(k)
                        for k in ("symptoms", "key_concepts", "patient_age", "duration")
                    },
                    ensure_ascii=False,
                ),
            )
        )
        q = str((data or {}).get("query", "")).strip()
        if q:
            return q
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] query generation failed ({condition}): {exc}")
    return condition


def scan(clinical: dict[str, Any], hypotheses: list[dict]) -> dict[str, Any]:
    records: list[dict] = []
    seen: set[str] = set()
    queries: list[dict] = []
    per_condition = max(2, config.WEB_MAX_RECORDS // max(1, len(hypotheses) or 1))

    for h in hypotheses[:3]:
        condition = h.get("condition", "").strip()
        if not condition:
            continue
        q = _build_query(condition, clinical)
        queries.append({"condition": condition, "query": q})
        records_found = pubmed.search(q, retmax=per_condition)
        if not records_found and q.strip() != condition.strip():
            queries[-1]["fallback"] = f"retry with '{condition}'"
            records_found = pubmed.search(condition, retmax=per_condition)
        for rec in records_found:
            if rec["pmid"] in seen:
                continue
            seen.add(rec["pmid"])
            level, label = pubmed.evidence_level(rec.get("pubtypes", []))
            rec = {
                **rec,
                "evidence_level": level,
                "evidence_label": label,
                "query_used": q,
                "condition": condition,
                "weight": round(1.0 / level, 3),
            }
            records.append(rec)

    records.sort(key=lambda r: (r["evidence_level"], r["year"]), reverse=False)
    return {
        "queries": queries,
        "records": records,
        "search_failures": [] if records else ["no records retrieved"],
    }
