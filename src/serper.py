"""Serper.dev web-search client (paper Sec. II-A-3).

The paper sends programmatic searches per hypothesis through "a web search API
like Serper" to scholarly indexes and settled guideline databases. This module
is the Serper half; NICE has its own keyless client (src/nice.py) and PubMed
already existed (src/pubmed.py). Google Scholar is deliberately NOT implemented
(see README deviations: no official API, scraping violates ToS).

Safety rules:
- the API key is read from .env via config and is never logged, printed or
  written to any file;
- every failure returns ([], reason) instead of raising, so one broken source
  can never take down the evidence stream (the paper's failure isolation);
- nothing is ever invented: an empty result is reported as an empty result.

Reproducibility: responses are disk-cached (results/serper_cache, gitignored)
so identical queries cost no credits and reruns return the same pages.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time

import requests

from . import config

ENDPOINT = "https://google.serper.dev/search"
_lock = threading.Lock()
_last_call = 0.0
MIN_INTERVAL = 0.4  # be polite to the API; the free tier is not rate-limit sensitive


def available() -> bool:
    """True only when a key is present in .env and SERPER_ENABLED is not 0."""
    return config.SERPER_ENABLED


def _throttle(min_interval: float = MIN_INTERVAL) -> None:
    global _last_call
    with _lock:
        wait = min_interval - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()


def _cache_path(query: str, num: int):
    digest = hashlib.sha256(f"{query}|{num}".encode("utf-8")).hexdigest()[:32]
    return config.SERPER_CACHE_DIR / f"{digest}.json"


def _cache_get(query: str, num: int) -> list[dict] | None:
    path = _cache_path(query, num)
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            records = data.get("records")
            if isinstance(records, list):
                return records
    except Exception:  # noqa: BLE001 - corrupt cache is simply refetched
        pass
    return None


def _cache_put(query: str, num: int, records: list[dict], credits) -> None:
    try:
        config.SERPER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(query, num).write_text(
            json.dumps(
                {
                    "query": query,
                    "num": num,
                    "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "credits_remaining": credits,
                    "records": records,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001 - caching is best effort
        print(f"[warn] serper cache write failed: {exc}")


def _normalise(payload: dict, num: int) -> list[dict]:
    records: list[dict] = []
    for item in payload.get("organic", [])[:num]:
        link = str(item.get("link") or "").strip()
        title = str(item.get("title") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        if not link or not title:
            continue
        records.append(
            {
                "title": title,
                "url": link,
                "link": link,
                # `abstract` is what Fusion displays / token-matches on, so web
                # pages expose their snippet under the same key PubMed uses.
                "abstract": snippet,
                "snippet": snippet,
                "source": "web",
                "year": "",
                "journal": "",
                "pubtypes": [],
                "pmid": "",
            }
        )
    return records


def search(query: str, num: int = 5) -> tuple[list[dict], str | None]:
    """Search the web via Serper. Returns (records, error_reason).

    Never raises; never fabricates. error_reason is None on success.
    """
    if not config.SERPER_ENABLED:
        return [], "serper_disabled_no_key"
    q = " ".join(str(query or "").split())
    if not q:
        return [], "serper_empty_query"

    cached = _cache_get(q, num)
    if cached is not None:
        return cached, None

    try:
        _throttle()
        response = requests.post(
            ENDPOINT,
            headers={
                "X-API-KEY": config.SERPER_API_KEY,  # from .env, never logged
                "Content-Type": "application/json",
            },
            json={"q": q, "num": num},
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001 - isolate the source
        return [], f"serper_request_failed: {exc}"

    if response.status_code == 401:
        return [], "serper_unauthorized_401_check_SERPER_API_KEY"
    if response.status_code == 402:
        return [], "serper_out_of_credits_402"
    if response.status_code == 429:
        return [], "serper_rate_limited_429"
    try:
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001
        return [], f"serper_bad_response: {exc}"

    records = _normalise(payload, num)
    _cache_put(q, num, records, payload.get("credits"))
    return records, None
