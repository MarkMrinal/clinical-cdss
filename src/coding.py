"""Stage 4 of the Clinical Text Clarifier - terminology mapping (paper II-A-1).

The paper maps every identified clinical concept to SNOMED CT or ICD-10 codes.
This module implements the ICD-10-CM half with the free NLM Clinical Tables API
(no API key, no cost): https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search

Honesty rules:
  * Only codes actually returned by the NLM API are reported. Nothing is
    fabricated when no match exists - the concept is listed in
    ``unmapped_concepts`` with a reason.
  * SNOMED CT is deferred (licence/membership required) and is never claimed.
  * ICD-10-CM codes diagnoses and symptoms, not drugs or lab tests, so
    Medication / Test Result entities are recorded as unmapped with that reason.
  * Any network/API failure degrades to an empty result with the reason
    recorded - it never crashes the pipeline.
"""
from __future__ import annotations

import json
import re
import threading
import time
from typing import Any

import requests

from . import config

API = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"
CODED_TYPES = {"Symptom", "Disease"}  # ICD-10-CM covers these; not meds/tests
TYPE_SKIP_REASON = {
    "Medication": "ICD-10-CM does not code medications (RxNorm/SNOMED territory)",
    "Test Result": "ICD-10-CM does not code test results (LOINC/SNOMED territory)",
    "Other": "entity class outside the four paper classes - not mapped",
}

_lock = threading.Lock()
_cache: dict[str, Any] | None = None
_last_call = 0.0


def _cache_path():
    return config.RAW_DIR / "icd10cm_cache.json"


def _load_cache() -> dict[str, Any]:
    global _cache
    with _lock:
        if _cache is None:
            try:
                _cache = json.loads(_cache_path().read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - missing/corrupt cache is fine
                _cache = {}
        return _cache


def _save_cache() -> None:
    with _lock:
        if _cache is None:
            return
        try:
            _cache_path().write_text(
                json.dumps(_cache, ensure_ascii=False, indent=1), encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001 - cache write failure is not fatal
            print(f"[warn] icd10cm cache not saved: {exc}")


def _throttle(min_interval: float = 0.2) -> None:
    global _last_call
    with _lock:
        wait = min_interval - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()


def _clean_term(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9 \-]", " ", str(text))).strip()


def _search(term: str) -> list[list[str]]:
    """One NLM call. Returns [[code, description], ...] (possibly empty)."""
    _throttle()
    r = requests.get(
        API, params={"terms": term, "sf": "code,name", "maxList": 25}, timeout=(10, 25)
    )
    r.raise_for_status()
    payload = r.json()
    rows = payload[3] if isinstance(payload, list) and len(payload) > 3 else []
    return [row for row in rows if isinstance(row, list) and len(row) >= 2]


def _best_match(term: str, rows: list[list[str]]) -> tuple[str, str] | None:
    """Pick the best row locally: the API returns code-ordered results, so we
    re-rank by (description starts with the term) > token overlap > shorter
    description > original position. E.g. 'pneumonia' -> J18.9 'Pneumonia,
    unspecified organism' instead of the alphabetically-first oddity."""
    tl = term.lower()
    t_tokens = set(re.findall(r"[a-z0-9]+", tl))
    best: tuple[str, str] | None = None
    best_key: tuple[float, float, int, int] | None = None
    for i, row in enumerate(rows):
        code, name = str(row[0]), str(row[1])
        nl = name.lower()
        n_tokens = set(re.findall(r"[a-z0-9]+", nl))
        overlap = len(t_tokens & n_tokens) / max(1, len(t_tokens))
        key = (1.0 if nl.startswith(tl) else 0.0, overlap, -len(n_tokens), -i)
        if best_key is None or key > best_key:
            best, best_key = (code, name), key
    return best


def _fallback_terms(term: str) -> list[str]:
    """When the full phrase matches nothing (ICD-10-CM descriptions rarely contain
    phrases like 'community acquired pneumonia'), retry with its most significant
    single tokens, longest first (head nouns like 'pneumonia' are usually long)."""
    tokens = {t for t in re.findall(r"[A-Za-z0-9\-]+", term) if len(t) >= 4}
    return sorted(tokens, key=len, reverse=True)


def _lookup(term: str) -> dict[str, Any]:
    """Cached lookup: {'code','display','matched_on'} | {'none': True} |
    {'error': msg}. Errors are never cached (they are transient)."""
    cache = _load_cache()
    key = term.lower()
    if key in cache:
        return cache[key]
    result: dict[str, Any] | None = None
    for cand in [term, *_fallback_terms(term)[:3]]:
        try:
            rows = _search(cand)
        except Exception as exc:  # noqa: BLE001 - never crash the pipeline
            return {"error": str(exc)[:200]}  # uncached so a later run can retry
        best = _best_match(cand, rows) if rows else None
        if best:
            result = {"code": best[0], "display": best[1], "matched_on": cand}
            break
    if result is None:
        result = {"none": True}
    with _lock:
        cache[key] = result
    _save_cache()
    return result


def map_concepts(entities: list[dict[str, Any]]) -> tuple[list[dict], list[dict], str]:
    """Map NER entities to ICD-10-CM codes.

    Returns (codes, unmapped, status) where each code entry is
    {"text", "system": "ICD-10-CM", "code", "display"} and each unmapped entry
    is {"text", "reason"}. ``status`` is "ok", "disabled" or "degraded".
    """
    codes: list[dict] = []
    unmapped: list[dict] = []

    if not config.ICD10_ENABLED:
        return codes, [
            {"text": e.get("text", ""), "reason": "ICD-10-CM mapping disabled (ICD10_ENABLED=0)"}
            for e in entities
        ], "disabled"

    errors = 0
    queried = 0
    for ent in entities:
        text = str(ent.get("text") or "").strip()
        etype = str(ent.get("type") or "")
        if not text:
            continue
        if etype not in CODED_TYPES:
            unmapped.append(
                {"text": text, "reason": TYPE_SKIP_REASON.get(etype, "entity type not mapped")}
            )
            continue
        if queried >= config.ICD10_MAX_QUERIES:
            unmapped.append({"text": text, "reason": "per-case ICD-10-CM query budget reached"})
            continue
        queried += 1
        result = _lookup(text)
        if "code" in result:
            codes.append(
                {
                    "text": text,
                    "system": "ICD-10-CM",
                    "code": result["code"],
                    "display": result.get("display", ""),
                    "matched_on": result.get("matched_on", text),
                }
            )
        elif "error" in result:
            errors += 1
            unmapped.append({"text": text, "reason": f"NLM API error: {result['error']}"})
        else:
            unmapped.append({"text": text, "reason": "no ICD-10-CM match found"})

    status = "degraded" if errors else "ok"
    return codes, unmapped, status
