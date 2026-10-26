"""Thin NCBI E-utilities client shared by the bulk fetcher and the Evidence Scanner."""
from __future__ import annotations

import threading
import time
import xml.etree.ElementTree as ET

import requests

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL = "clinical-cdss-replication"
_lock = threading.Lock()
_last_call = 0.0


def _throttle(min_interval: float = 0.35) -> None:
    global _last_call
    with _lock:
        wait = min_interval - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()


def esearch(term: str, retmax: int = 100) -> list[str]:
    _throttle()
    r = requests.get(
        f"{BASE}/esearch.fcgi",
        params={
            "db": "pubmed",
            "term": term,
            "retmax": retmax,
            "retmode": "json",
            "sort": "relevance",
            "tool": TOOL,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["esearchresult"].get("idlist", [])


def efetch(pmids: list[str]) -> list[dict]:
    if not pmids:
        return []
    _throttle()
    r = requests.post(
        f"{BASE}/efetch.fcgi",
        data={"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "tool": TOOL},
        timeout=60,
    )
    r.raise_for_status()
    root = ET.fromstring(r.content)
    out: list[dict] = []
    for art in root.findall(".//PubmedArticle"):
        pmid = (art.findtext(".//PMID") or "").strip()
        title_node = art.find(".//ArticleTitle")
        title = "".join(title_node.itertext()).strip() if title_node is not None else ""
        parts = []
        for node in art.findall(".//Abstract/AbstractText"):
            label = node.get("Label")
            txt = "".join(node.itertext()).strip()
            parts.append(f"{label}: {txt}" if label else txt)
        abstract = " ".join(parts).strip()
        if not pmid or not abstract:
            continue
        out.append(
            {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "journal": art.findtext(".//Journal/Title") or "",
                "year": art.findtext(".//JournalIssue/PubDate/Year")
                or art.findtext(".//ArticleDate/Year")
                or "",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "pubtypes": [n.text for n in art.findall(".//PublicationType") if n.text],
            }
        )
    return out


def search(term: str, retmax: int = 5) -> list[dict]:
    try:
        return efetch(esearch(f"{term} AND hasabstract[text]", retmax=retmax))
    except Exception as exc:  # noqa: BLE001 - never fabricate evidence
        print(f"[warn] PubMed search failed for {term!r}: {exc}")
        return []


LEVEL_BY_PUBTYPE = {
    "systematic review": 1,
    "meta-analysis": 1,
    "practice guideline": 1,
    "guideline": 1,
    "randomized controlled trial": 2,
    "review": 2,
    "clinical trial": 2,
    "observational study": 3,
    "case reports": 4,
}


def evidence_level(pubtypes: list[str]) -> tuple[int, str]:
    """Paper II-A-3/II-C: weight systematic reviews and guidelines above case reports."""
    lowered = [p.lower() for p in pubtypes]
    for label, level in LEVEL_BY_PUBTYPE.items():
        if any(label in p for p in lowered):
            return level, label
    return 3, "unspecified"
