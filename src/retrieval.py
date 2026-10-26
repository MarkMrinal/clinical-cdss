"""BioBERT embeddings + Qdrant vector search (paper Sec. II-B)."""
from __future__ import annotations

import functools
import threading
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


def _raw_search(vector: list[float], top_k: int) -> list[dict]:
    client = _get_client()
    per_collection: dict[str, list] = {}
    with _qdrant_lock:
        for name in config.COLLECTIONS:
            per_collection[name] = _search_collection(client, name, vector, top_k)

    merged: dict[str, dict] = {}
    best_by_collection: dict[str, str] = {}
    for name, pts in per_collection.items():
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
            }
        )
    return out


def search(query: str, top_k: int | None = None) -> list[dict]:
    top_k = top_k or config.TOP_K
    return _raw_search(embed_one(query), top_k)


def search_with_fallback(
    query: str, simplified_query: str | None = None, top_k: int | None = None
) -> tuple[list[dict], dict]:
    """Paper II-B-2: three-level fallback (lower threshold -> simplify -> widen k)."""
    top_k = top_k or config.TOP_K
    trace = {"query": query, "levels": []}

    hits = _raw_search(embed_one(query), top_k)
    best = max((h["score"] for h in hits), default=0.0)
    trace["levels"].append({"level": 1, "strategy": "exact_rrf", "best_score": best})

    if best < 0.50:
        hits = _raw_search(embed_one(query), top_k * 2)
        best = max((h["score"] for h in hits), default=0.0)
        trace["levels"].append(
            {"level": 2, "strategy": "widened_k", "best_score": best}
        )
        hits = hits[:top_k]

    if best < 0.40 and simplified_query:
        hits = _raw_search(embed_one(simplified_query), top_k)
        best = max((h["score"] for h in hits), default=0.0)
        trace["levels"].append(
            {"level": 3, "strategy": "simplified", "best_score": best}
        )

    return hits, trace
