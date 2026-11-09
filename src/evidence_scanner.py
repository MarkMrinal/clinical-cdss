"""AGENT 3 - Evidence-Based Scanner (paper II-A-3).

Per hypothesis it fans out to three sources, exactly as the paper describes
("programmatic searches per hypothesis ... to PubMed, Google Scholar, and
settled clinical guideline databases like NICE"):

  1. PubMed        - NCBI E-utilities (keyless) with evidence-level tagging
  2. NICE          - nice.org.uk guideline search (keyless, src/nice.py)
  3. Web (Serper)  - general web results (src/serper.py, key from .env)

Google Scholar is deliberately not implemented (no official API; ToS).

Failure isolation: each source is wrapped independently and reports into
`sources` / `search_failures`, so a dead Serper key or a NICE layout change
degrades to fewer sources instead of breaking the case. Evidence is never
fabricated on failure.

Optionally (SCANNER_WEB_SUMMARY) one LLM pass turns the returned web pages into
short per-hypothesis summaries - the paper's "derives text summaries pertinent
to each hypothesis". If it fails, the raw snippets are used unchanged.
"""
from __future__ import annotations

import json
from typing import Any

from . import config, nice, pubmed, serper
from .llm import generate_json

QUERY_PROMPT = """You are the Evidence-Based Scanner of a clinical decision support system.
Write ONE PubMed search query (max 12 words, no boolean operators except AND) that would find
the strongest current evidence for this candidate condition, given the patient context.

Return ONLY JSON: {{"query": "..."}}

Candidate condition: {condition}
Patient context: {clinical}
"""

SUMMARY_PROMPT = """You are the Evidence-Based Scanner of a clinical decision support system.
You are given external web evidence (NICE guidelines and web pages) retrieved for several
candidate conditions. Write a 2-3 sentence summary PER condition that only uses facts present
in the listed pages, and cite the pages you used as [W1], [W2], ... (their numbers below).
Do not add outside knowledge. If a condition has no web evidence, omit it.

Return ONLY JSON: {{"summaries": [{{"condition": "...", "summary": "..."}}]}}

EXTERNAL WEB EVIDENCE:
{records}
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
            ),
            agent="evidence_scanner",
        )
        q = str((data or {}).get("query", "")).strip()
        if q:
            return q
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] query generation failed ({condition}): {exc}")
    return condition


def _key(rec: dict) -> str:
    """Dedup key across sources: PMID when we have one, else the URL."""
    return str(rec.get("pmid") or rec.get("url") or rec.get("title") or "")


def _finish(rec: dict, q: str, condition: str, level: int, label: str) -> dict:
    return {
        **rec,
        # nice.py / serper.py tag their own records; PubMed records arrive
        # untagged, so the shared schema defaults to the PubMed source
        "source": rec.get("source") or "pubmed",
        "evidence_level": level,
        "evidence_label": label,
        "query_used": q,
        "condition": condition,
        "weight": round(1.0 / level, 3),
    }


def _summarise_web(records: list[dict]) -> list[dict]:
    """Paper II-A-3: derive per-hypothesis summaries from the returned web pages.

    One LLM call for the whole scan. Returns [] when disabled, when there is
    nothing to summarise, or when the model fails (raw snippets stay in use).
    """
    web_records = [r for r in records if r.get("source") in ("web", "nice")]
    if not config.SCANNER_WEB_SUMMARY or not web_records:
        return []
    listing = "\n".join(
        f"[W{i}] (source: {r.get('source')}, level {r.get('evidence_level')}, "
        f"condition: {r.get('condition')}) {r.get('title', '')} - "
        f"{str(r.get('abstract', ''))[:400]}"
        for i, r in enumerate(records, 1)
        if r.get("source") in ("web", "nice")
    )
    try:
        data = generate_json(
            SUMMARY_PROMPT.format(records=listing), agent="evidence_scanner"
        )
    except Exception as exc:  # noqa: BLE001 - summaries are optional
        print(f"[warn] web summarisation failed: {exc}")
        return []
    out: list[dict] = []
    for item in (data or {}).get("summaries", []):
        cond = str(item.get("condition", "")).strip()
        summary = str(item.get("summary", "")).strip()
        if cond and summary:
            out.append({"condition": cond, "summary": summary})
    return out


def scan(clinical: dict[str, Any], hypotheses: list[dict]) -> dict[str, Any]:
    records: list[dict] = []
    seen: set[str] = set()
    queries: list[dict] = []
    failures: list[str] = []
    sources: dict[str, str] = {
        "pubmed": "ok" if hypotheses else "skipped_no_hypotheses",
        "nice": (
            "disabled" if not nice.available() else "ok" if hypotheses else "skipped_no_hypotheses"
        ),
        "serper": (
            "disabled_no_key"
            if not serper.available()
            else "ok" if hypotheses else "skipped_no_hypotheses"
        ),
    }
    per_condition = max(2, config.WEB_MAX_RECORDS // max(1, len(hypotheses) or 1))

    for h in hypotheses[:3]:
        condition = h.get("condition", "").strip()
        if not condition:
            continue
        q = _build_query(condition, clinical)
        queries.append({"condition": condition, "query": q})

        # --- source 1: PubMed (existing behaviour, now isolated) --------------
        try:
            records_found = pubmed.search(q, retmax=per_condition)
            if not records_found and q.strip() != condition.strip():
                queries[-1]["fallback"] = f"retry with '{condition}'"
                records_found = pubmed.search(condition, retmax=per_condition)
            if not records_found:
                failures.append(f"pubmed: no records for {condition!r}")
            for rec in records_found:
                k = _key(rec)
                if k in seen:
                    continue
                seen.add(k)
                level, label = pubmed.evidence_level(rec.get("pubtypes", []))
                records.append(_finish(rec, q, condition, level, label))
        except Exception as exc:  # noqa: BLE001 - isolate the source
            failures.append(f"pubmed: {exc}")
            sources["pubmed"] = "failed"

        # --- source 2: NICE guidelines (keyless) -----------------------------
        if nice.available():
            try:
                nice_records, err = nice.search(condition, num=2)
                if err:
                    failures.append(f"nice: {err}")
                    if sources["nice"] == "ok":
                        sources["nice"] = "no_results"
                for rec in nice_records:
                    k = _key(rec)
                    if k in seen:
                        continue
                    seen.add(k)
                    # nice.py pre-classifies its result types; take those values
                    # before _finish re-attaches them under the shared schema
                    level = rec.pop("evidence_level", 3)
                    label = rec.pop("evidence_label", "nice guidance")
                    records.append(
                        _finish(rec, f"NICE search: {condition}", condition, level, label)
                    )
            except Exception as exc:  # noqa: BLE001
                failures.append(f"nice: {exc}")
                sources["nice"] = "failed"

        # --- source 3: Serper web search (needs .env key) --------------------
        if serper.available():
            try:
                web_records, err = serper.search(condition, num=per_condition)
                if err:
                    failures.append(f"serper: {err}")
                    if sources["serper"] == "ok":
                        sources["serper"] = "failed"
                for rec in web_records:
                    k = _key(rec)
                    if k in seen:
                        continue
                    seen.add(k)
                    # Unverifiable web pages rank below case reports; a link to
                    # PubMed/PMC is known literature but its design is unknown.
                    url = str(rec.get("url", "")).lower()
                    if "pubmed.ncbi.nlm.nih.gov" in url or "pmc.ncbi.nlm.nih.gov" in url:
                        level, label = 3, "web_literature"
                    else:
                        level, label = 4, "web_page"
                    records.append(_finish(rec, q, condition, level, label))
            except Exception as exc:  # noqa: BLE001
                failures.append(f"serper: {exc}")
                sources["serper"] = "failed"

    # best evidence first (level 1 = guideline/meta-analysis), then newest-ish
    records.sort(key=lambda r: (r.get("evidence_level", 3), str(r.get("year") or "")))

    if not records:
        failures.append("no records retrieved from any source")

    return {
        "queries": queries,
        "records": records,
        "sources": sources,
        "search_failures": failures,
        "web_summaries": _summarise_web(records),
    }
