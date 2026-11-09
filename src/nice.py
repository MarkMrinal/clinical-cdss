"""NICE guideline search - keyless (paper Sec. II-A-3).

The paper feeds per-hypothesis searches through a web-search API to "settled
clinical guideline databases like NICE" (National Institute for Health and Care
Excellence). NICE publishes its search results page without any key or quota,
so this client queries it directly: guideline evidence keeps working even when
no SERPER_API_KEY is configured (graceful degradation, no credits spent).

Safety rules (same as src/serper.py): every failure returns ([], reason) rather
than raising, and nothing is ever invented.

The HTML layout is parsed defensively (BEM-ish card classes, `headinglink`
attribute) and every field is optional: if NICE changes its markup the parse
yields zero results plus a recorded reason instead of a crash.
"""
from __future__ import annotations

import html
import re
import threading
import time

import requests

from . import config

SEARCH_URL = "https://www.nice.org.uk/search"
USER_AGENT = "clinical-cdss-replication/1.0 (academic replication; local)"
_lock = threading.Lock()
_last_call = 0.0
MIN_INTERVAL = 0.6  # one request per 0.6 s at most; NICE is a public site

# Each result lives in <li class="...SearchCardList_listItem..."> ... </li>.
# The hashed suffix may change with NICE deployments, so match on the stable
# prefix and split on it (nested <li> inside the card use a different class).
_LI_SPLIT = re.compile(r'<li class="[^"]*SearchCardList_listItem[^"]*"')
_TAG = re.compile(r"<[^>]+>")
_HEADING = re.compile(
    r'card__heading[^>]*>\s*<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>', re.S
)
_SUMMARY = re.compile(r'card__summary[^>]*>\s*<span[^>]*>(.*?)</span>', re.S)
_HEADINGLINK = re.compile(r'headinglink="([^"]+)"')
_TYPE = re.compile(r"<dt[^>]*>\s*Result type\s*</dt>\s*<dd>(.*?)</dd>", re.S)
_DATE = re.compile(
    r"<dt[^>]*>\s*(?:Published|Last updated)\s*</dt>\s*<dd>(.*?)</dd>", re.S
)
_YEAR = re.compile(r"\b(19|20)\d{2}\b")

# NICE result types mapped onto the evidence hierarchy the paper describes
# ("guidelines/systematic reviews > case reports"). Types we cannot classify
# stay at level 3 (unspecified) instead of being promoted.
_FORMAL_LEVEL1 = ("guideline", "quality standard", "technology appraisal")
_EVIDENCE_LEVEL2 = ("evidence summary", "advice", "research recommendation")


def available() -> bool:
    return config.NICE_ENABLED


def _throttle(min_interval: float = MIN_INTERVAL) -> None:
    global _last_call
    with _lock:
        wait = min_interval - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()


def _clean(fragment: str) -> str:
    """Strip tags, unescape entities, collapse whitespace."""
    return " ".join(html.unescape(_TAG.sub(" ", fragment)).split())


def evidence_level(result_type: str) -> tuple[int, str]:
    """Weight NICE products the way the paper weights guidelines."""
    low = (result_type or "").lower()
    if any(k in low for k in _FORMAL_LEVEL1):
        return 1, "guideline"
    if any(k in low for k in _EVIDENCE_LEVEL2):
        return 2, "evidence summary"
    return 3, "nice guidance"


def search(query: str, num: int = 5) -> tuple[list[dict], str | None]:
    """Search nice.org.uk. Returns (records, error_reason). Never raises."""
    if not config.NICE_ENABLED:
        return [], "nice_disabled"
    q = " ".join(str(query or "").split())
    if not q:
        return [], "nice_empty_query"

    try:
        _throttle()
        response = requests.get(
            SEARCH_URL,
            params={"q": q},
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - isolate the source
        return [], f"nice_request_failed: {exc}"

    blocks = _LI_SPLIT.split(response.text)[1:]  # [0] is the page preamble
    records: list[dict] = []
    for block in blocks:
        if len(records) >= num:
            break
        m_head = _HEADING.search(block)
        if not m_head:
            continue
        href, title_raw = m_head.group(1), m_head.group(2)
        m_link = _HEADINGLINK.search(block)
        path = m_link.group(1) if m_link else href
        if not path:
            continue
        url = path if path.startswith("http") else f"https://www.nice.org.uk{path}"

        m_sum = _SUMMARY.search(block)
        m_type = _TYPE.search(block)
        m_date = _DATE.search(block)
        result_type = _clean(m_type.group(1)) if m_type else "NICE"
        date = _clean(m_date.group(1)) if m_date else ""
        year_m = _YEAR.search(date)
        level, label = evidence_level(result_type)

        records.append(
            {
                "title": _clean(title_raw),
                "url": url,
                "link": url,
                "abstract": _clean(m_sum.group(1)) if m_sum else "",
                "snippet": _clean(m_sum.group(1)) if m_sum else "",
                "source": "nice",
                "year": year_m.group(0) if year_m else "",
                "journal": "NICE",
                "pubtypes": [result_type],
                "pmid": "",
                "nice_type": result_type,
                "nice_date": date,
                # pre-computed so the scanner does not re-classify
                "evidence_level": level,
                "evidence_label": label,
            }
        )

    if not records:
        return [], "nice_no_results_or_layout_changed"
    return records, None
