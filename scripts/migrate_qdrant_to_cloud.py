"""Copy this project's local Qdrant collections to Qdrant Cloud.

Set QDRANT_URL and QDRANT_API_KEY in the project .env, then run:
    python scripts/migrate_qdrant_to_cloud.py

The script preserves point IDs, vectors, and payloads. Existing destination
collections are updated by re-uploading the same IDs, so rerunning can resume
an interrupted migration.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src import config  # noqa: E402

BATCH_SIZE = 128


def main() -> None:
    from qdrant_client import QdrantClient

    url = os.getenv("QDRANT_URL", "").strip()
    api_key = os.getenv("QDRANT_API_KEY", "").strip()
    if not url or not api_key:
        raise SystemExit(
            "Set QDRANT_URL and QDRANT_API_KEY in the project .env before migrating."
        )

    source = QdrantClient(path=str(config.INDEX_DIR))
    destination = QdrantClient(url=url, api_key=api_key)

    for name in config.COLLECTIONS:
        if not source.collection_exists(name):
            raise SystemExit(
                f"Local collection '{name}' was not found in {config.INDEX_DIR}"
            )

        source_info = source.get_collection(name)
        if not destination.collection_exists(name):
            destination.create_collection(
                collection_name=name,
                vectors_config=source_info.config.params.vectors,
                shard_number=1,
            )

        expected = source_info.points_count or 0
        print(f"Copying '{name}' ({expected:,} points)...")
        offset = None
        copied = 0
        while True:
            points, offset = source.scroll(
                collection_name=name,
                limit=BATCH_SIZE,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            if points:
                destination.upload_points(
                    collection_name=name,
                    points=points,
                    batch_size=BATCH_SIZE,
                    parallel=1,
                    max_retries=3,
                )
                copied += len(points)
                print(f"  {copied:,}/{expected:,}", end="\r", flush=True)
            if offset is None:
                break

        remote_count = destination.count(name, exact=True).count
        print(f"\n  Cloud collection '{name}': {remote_count:,} points")
        if remote_count < expected:
            raise SystemExit(
                f"Migration incomplete for '{name}': expected at least {expected:,}, "
                f"found {remote_count:,}. You can rerun the script to resume."
            )

    print("Migration complete. The local index was not changed.")


if __name__ == "__main__":
    main()
