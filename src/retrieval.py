"""BioBERT embeddings + Qdrant vector search (paper Sec. II-B)."""
from __future__ import annotations

import functools
import re
import threading
import time
from typing import Any

from . import config

_encoder: Any = None
_client: Any = None
_qdrant_lock = threading.Lock()  # local Qdrant client is not thread-safe
_client_lock = threading.Lock()  # ...and neither is opening it (embedded mode is single-writer)
_encoder_lock = threading.Lock()  # 4 eval threads must not each load a 440 MB model
_encode_lock = threading.Lock()   # serialise CPU encode; concurrency only adds peak RAM


def _get_encoder():
    global _encoder
    if _encoder is None:
        with _encoder_lock:
            if _encoder is None:
                from sentence_transformers import SentenceTransformer

                _encoder = SentenceTransformer(config.EMBED_MODEL, device="cpu")
    return _encoder


def _get_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                from qdrant_client import QdrantClient

                if config.QDRANT_URL:
                    if not config.QDRANT_API_KEY:
                        raise RuntimeError(
                            "QDRANT_API_KEY is required when QDRANT_URL is set"
                        )
                    _client = QdrantClient(
                        url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY
                    )
                else:
                    _client = QdrantClient(path=str(config.INDEX_DIR))
    return _client


def embed(texts: list[str]) -> list[list[float]]:
    model = _get_encoder()
    with _encode_lock:
        vecs = model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    return [v.tolist() for v in vecs]


def embed_one(text: str) -> list[float]:
    return embed([text])[0]


def _search_collection(client, name: str, vector: list[float], top_k: int):
    try:
        try:
            # score_threshold is applied client-side below (version-proof and
            # works in embedded mode); we just over-fetch candidates here.
            res = client.query_points(
                collection_name=name, query=vector, limit=top_k, with_payload=True
            )
            return res.points
        except AttributeError:  # older qdrant-client
            return client.search(
                collection_name=name, query_vector=vector, limit=top_k, with_payload=True
            )
    except Exception as exc:  # noqa: BLE001 - log, never fail silently
        print(f"[warn] retrieval failed on collection '{name}': {exc}")
        return []


def _rrf(rank: int, k: int = 60) -> float:
    """Reciprocal rank fusion: comparable across collections with different score scales."""
    return 1.0 / (k + rank)


def _raw_search(
    vector: list[float], top_k: int, score_threshold: float | None = None
) -> list[dict]:
    client = _get_client()
    # Over-fetch when a threshold is active so filtering cannot starve top_k.
    fetch_k = top_k if score_threshold is None else min(top_k * 5, 100)
    per_collection: dict[str, list] = {}
    with _qdrant_lock:
        for name in config.COLLECTIONS:
            per_collection[name] = _search_collection(client, name, vector, fetch_k)

    merged: dict[str, dict] = {}
    best_by_collection: dict[str, str] = {}
    for name, pts in per_collection.items():
        # Paper II-B-2 / Table 2: passages below the similarity threshold tau
        # are not admissible evidence. Filtering is done client-side so it works
        # identically in embedded and cloud Qdrant.
        if score_threshold is not None:
            pts = [p for p in pts if float(p.score) >= score_threshold]
        if pts:
            best_by_collection[name] = str(pts[0].id)
        for rank, p in enumerate(pts, 1):
            pid = str(p.id)
            entry = merged.setdefault(pid, {"point": p, "rrf": 0.0, "collections": []})
            entry["rrf"] += _rrf(rank)
            entry["collections"].append(name)

    ordered = sorted(merged.values(), key=lambda e: e["rrf"], reverse=True)

    # Guarantee one passage per collection so the fusion stage always receives two
    # independent evidence streams (paper Sec. II-A-4), then fill by RRF.
    chosen: list[str] = []
    for pid in best_by_collection.values():
        if pid not in chosen:
            chosen.append(pid)
    for e in ordered:
        if len(chosen) >= top_k:
            break
        pid = str(e["point"].id)
        if pid not in chosen:
            chosen.append(pid)

    entries = [merged[pid] for pid in chosen[:top_k]]
    entries.sort(key=lambda e: e["rrf"], reverse=True)

    out: list[dict] = []
    for e in entries:
        p = e["point"]
        pl = p.payload or {}
        out.append(
            {
                "chunk_id": str(p.id),
                "score": float(p.score),
                "rrf": round(e["rrf"], 5),
                "collections": e["collections"],
                "text": pl.get("text", ""),
                "title": pl.get("title", ""),
                "url": pl.get("url", ""),
                "source": pl.get("source", ""),
                "topic": pl.get("topic", ""),
                "pmid": pl.get("pmid", ""),
                # paper Table 2 / Sec. II-B-2: evidence carries document info
                "length": pl.get("length", len(pl.get("text", ""))),
            }
        )
    return out


def search(
    query: str, top_k: int | None = None, score_threshold: float | None = None
) -> list[dict]:
    top_k = top_k or config.TOP_K
    return _raw_search(embed_one(query), top_k, score_threshold=score_threshold)


def _boost_query(query: str, boost_terms: list[str] | None) -> str:
    """Paper Table 2 'term boosting: Active'.

    The paper never publishes the algorithm, so this is our documented
    approximation: append priority terms that are ABSENT from the query, repeat
    terms that are already present once (frequency boost), never boost negated
    terms ('no ...', 'denies', 'without'), at most 5 terms / 400 chars total.
    """
    if not boost_terms or not config.TERM_BOOST:
        return query
    q = query.lower()
    out = query
    added = 0
    seen: set[str] = set()
    for raw in boost_terms:
        term = " ".join(str(raw or "").split())
        low = term.lower()
        if len(term) < 3 or low in seen:
            continue
        if low.startswith(("no ", "denies", "without", "negative for", "not ")):
            continue  # never boost negations (paper: 'no ...' terms stay out)
        if added >= 5 or len(out) + len(term) + 1 > 400:
            break
        seen.add(low)
        out = f"{out} {term}"  # present terms get repeated, absent ones appended
        added += 1
    return out


def _strip_terms(query: str, terms: list[str] | None) -> str:
    """Remove symptom phrases from the query (Fig. 5, Strategy 3)."""
    out = query
    for raw in terms or []:
        term = " ".join(str(raw or "").split())
        if len(term) < 3:
            continue
        out = re.sub(rf"\b{re.escape(term)}\b", " ", out, flags=re.IGNORECASE)
    return " ".join(out.split())


def search_with_fallback(
    query: str,
    simplified_query: str | None = None,
    top_k: int | None = None,
    boost_terms: list[str] | None = None,
    symptom_terms: list[str] | None = None,
) -> tuple[list[dict], dict]:
    """Paper Fig. 5 retrieval loop with its ordered fallback strategies.

    boosting (Table 2) -> search at tau=0.65 -> Strategy 1 threshold reduction
    (0.55, then 0.45) -> Strategy 2 query simplification @0.45 -> Strategy 3
    remove symptoms @0.45 -> final no-threshold safety return (so the pipeline
    never receives zero passages).

    The rungs 0.65/0.55/0.45 come from Table 2 (tau=0.65) and Fig. 5
    annotations; they are NOT tuned against evaluation questions.
    """
    top_k = top_k or config.TOP_K
    t0 = time.perf_counter()
    tau = config.RETRIEVAL_THRESHOLD
    boosted = _boost_query(query, boost_terms)
    vec_cache: dict[str, list[float]] = {}
    trace: dict[str, Any] = {
        "query": query,
        "boosted_query": boosted,
        "levels": [],
        "retrieval_ms": 0.0,
    }

    def attempt(strategy: str, q: str, threshold: float | None) -> list[dict]:
        t = time.perf_counter()
        if q not in vec_cache:  # same query is tried at several rungs: embed once
            vec_cache[q] = embed_one(q)
        hits = _raw_search(vec_cache[q], top_k, score_threshold=threshold)
        trace["levels"].append(
            {
                "level": len(trace["levels"]) + 1,
                "strategy": strategy,
                "threshold": threshold,
                "n_hits": len(hits),
                "best_score": round(max((h["score"] for h in hits), default=0.0), 4),
                "query_used": q[:300],
                "ms": round((time.perf_counter() - t) * 1000, 1),
            }
        )
        return hits

    # 1) term-boosted search at the paper's threshold
    hits = attempt("boosted_search", boosted, tau)
    if hits:
        trace["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return hits, trace

    # Strategy 1 - threshold reduction (Fig. 5: 0.55 -> 0.45)
    for rung in (0.55, 0.45):
        hits = attempt(f"threshold_reduction@{rung}", boosted, rung)
        if hits:
            trace["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            return hits, trace

    # Strategy 2 - query simplification
    if simplified_query:
        hits = attempt("simplify_query", simplified_query, 0.45)
        if hits:
            trace["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            return hits, trace

    # Strategy 3 - remove symptom phrases from the query
    stripped = _strip_terms(boosted, symptom_terms)
    if stripped and stripped != boosted:
        hits = attempt("remove_symptoms", stripped, 0.45)
        if hits:
            trace["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            return hits, trace

    # Safety return - no threshold, so downstream stages always get evidence
    hits = attempt("no_threshold_safety", boosted, None)
    trace["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return hits, trace
