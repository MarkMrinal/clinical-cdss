"""Sync the local embedded Qdrant index -> Qdrant Cloud (idempotent).

  python scripts/sync_qdrant_cloud.py            # sync all 3 collections
  python scripts/sync_qdrant_cloud.py --only symptom_disease

Reads QDRANT_URL / QDRANT_API_KEY from .env (never printed). Copies every
point (vectors + payload) from data/raw/qdrant_index to the cloud so the
deployed Render service sees exactly the same 34,206-chunk index the local
runs use. Safe to re-run: upserts are keyed by point id.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

from qdrant_client import QdrantClient  # noqa: E402

PAGE = 1000       # scroll page size (local)
BATCH = 256       # upsert batch size (cloud)

from qdrant_client import models as qm  # noqa: E402


def _cfg_dict(cfg, cls):
    """Local (embedded) engine config models do not validate as the remote
    CreateCollection diff models - dump to a plain dict keeping only the
    fields the target model knows about (HnswConfig -> HnswConfigDiff)."""
    if cfg is None:
        return None
    data = cfg.model_dump() if hasattr(cfg, "model_dump") else dict(cfg)
    out = {k: v for k, v in data.items() if k in getattr(cls, "model_fields", {})}
    return out or None


def _to_points(records) -> list[dict]:
    return [
        {"id": r.id, "vector": r.vector, "payload": r.payload or {}}
        for r in records
    ]


def ensure_collection(cloud: QdrantClient, name: str, linfo) -> None:
    """Create the cloud collection with the LOCAL collection's exact config."""
    existing = {c.name for c in cloud.get_collections().collections}
    if name in existing:
        cinfo = cloud.get_collection(name)
        lv, cv = linfo.config.params.vectors, cinfo.config.params.vectors
        same = (
            getattr(lv, "size", None) == getattr(cv, "size", None)
            and getattr(lv, "distance", None) == getattr(cv, "distance", None)
        )
        if same:
            print(f"  [{name}] cloud collection exists, config matches - keeping")
            return
        print(f"  [{name}] cloud collection config MISMATCH - recreating from local")
        cloud.delete_collection(name)
    try:
        cloud.create_collection(
            collection_name=name,
            vectors_config=linfo.config.params.vectors,
            hnsw_config=_cfg_dict(linfo.config.hnsw_config, qm.HnswConfigDiff),
            on_disk_payload=linfo.config.params.on_disk_payload,
            quantization_config=linfo.config.quantization_config,
        )
    except Exception as exc:  # noqa: BLE001 - fall back to server defaults
        print(f"  [{name}] create with local tuning failed ({type(exc).__name__}); "
              f"retrying with server-default HNSW/quantization")
        cloud.create_collection(
            collection_name=name,
            vectors_config=linfo.config.params.vectors,
            on_disk_payload=linfo.config.params.on_disk_payload,
        )
    print(f"  [{name}] created on cloud with local config")


def sync_collection(local: QdrantClient, cloud: QdrantClient, name: str) -> int:
    linfo = local.get_collection(name)
    want = linfo.points_count
    print(f"[{name}] local points: {want}")
    ensure_collection(cloud, name, linfo)

    t0 = time.time()
    done = 0
    offset = None
    while True:
        records, offset = local.scroll(
            collection_name=name,
            limit=PAGE,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not records:
            break
        pts = _to_points(records)
        for i in range(0, len(pts), BATCH):
            cloud.upsert(
                collection_name=name,
                points=pts[i : i + BATCH],
                wait=True,
            )
        done += len(pts)
        print(f"  [{name}] {done}/{want} copied ({time.time() - t0:.0f}s)")
        if offset is None:
            break

    got = cloud.get_collection(name).points_count
    status = "OK" if got == want else "COUNT MISMATCH"
    print(f"  [{name}] cloud points: {got} (local {want}) -> {status}")
    if got != want:
        raise SystemExit(f"[{name}] sync incomplete: cloud {got} != local {want}")
    return got


def verify_payload(local: QdrantClient, cloud: QdrantClient, name: str) -> None:
    """Fetch the same id on both sides and diff the payload keys."""
    pts, _ = local.scroll(collection_name=name, limit=1, with_payload=True,
                          with_vectors=False)
    if not pts:
        return
    pid = pts[0].id
    lp = (local.retrieve(name, ids=[pid], with_payload=True)[0].payload) or {}
    cp = (cloud.retrieve(name, ids=[pid], with_payload=True)[0].payload) or {}
    missing = sorted(set(lp) - set(cp))
    extra = sorted(set(cp) - set(lp))
    print(f"  [{name}] payload spot-check id={pid}: "
          f"local keys={len(lp)} cloud keys={len(cp)} "
          f"missing_on_cloud={missing or '-'} extra={extra or '-'}")


def verify_search(local: QdrantClient, cloud: QdrantClient, name: str) -> None:
    """Sanity: searching cloud with a local vector must return that same id."""
    pts, _ = local.scroll(collection_name=name, limit=1, with_payload=False,
                          with_vectors=True)
    if not pts:
        return
    pid, vec = pts[0].id, pts[0].vector
    try:
        hits = cloud.query_points(name, query=vec, limit=3, with_payload=False).points
    except AttributeError:  # older qdrant-client
        hits = cloud.search(name, vector=vec, limit=3, with_payload=False)
    ok = any(h.id == pid for h in hits)
    print(f"  [{name}] search round-trip with source id: "
          f"{'OK' if ok else 'NOT IN TOP-3 (near-duplicate vector ok)'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="",
                    help="comma-separated collection name(s) to sync")
    args = ap.parse_args()

    url = config.QDRANT_URL
    key = config.QDRANT_API_KEY
    if not url:
        raise SystemExit("QDRANT_URL not set in .env - nothing to sync to")

    names = (
        [n.strip() for n in args.only.split(",") if n.strip()]
        if args.only
        else list(config.COLLECTIONS)
    )
    print(f"target: {url.split('//')[-1][:6]}... (key len {len(key)})")

    # Local side is ALWAYS the embedded index by path, even though QDRANT_URL
    # is now set (config would otherwise route this client to the cloud too).
    local = QdrantClient(path=str(config.INDEX_DIR))
    cloud = QdrantClient(url=url, api_key=key, timeout=120)

    total = 0
    for name in names:
        total += sync_collection(local, cloud, name)
        verify_payload(local, cloud, name)
        verify_search(local, cloud, name)

    print(f"\nDONE: {len(names)} collection(s), {total} points synced to cloud")


if __name__ == "__main__":
    main()
