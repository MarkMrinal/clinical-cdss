"""Second vector collection: medical textbooks (what the paper actually indexed).

Paper II-B-1 names "medical textbooks, clinical guidelines, epidemiological datasets".
Our PubMed-only index scored 0/20 on MedQA retrieval because exam items include basic
science that abstracts do not cover. This adds MedRAG/textbooks (18 standard texts,
125,847 passages) as a balanced sample: 780 passages per book.

We keep the textbooks' native paragraph segmentation instead of re-chunking to 512/64,
because they are already passage-sized (documented deviation).
"""
from __future__ import annotations

import random
import sys
import uuid
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.retrieval import embed  # noqa: E402

PER_BOOK = int(sys.argv[1]) if len(sys.argv) > 1 else 780
COLLECTION = "textbooks"


def main() -> None:
    from datasets import load_dataset
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    ds = load_dataset("MedRAG/textbooks", split="train")
    print(f"loaded {len(ds)} textbook passages")

    by_book: dict[str, list[int]] = {}
    for i, t in enumerate(ds["title"]):
        by_book.setdefault(t, []).append(i)

    random.seed(config.SAMPLE_SEED)
    selected: list[int] = []
    for book, idxs in sorted(by_book.items()):
        take = min(PER_BOOK, len(idxs))
        selected.extend(random.sample(idxs, take))
    random.shuffle(selected)
    print(f"selected {len(selected)} passages from {len(by_book)} books")

    client = QdrantClient(path=str(config.INDEX_DIR))
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

    existing: set[str] = set()
    offset = None
    while True:
        pts, offset = client.scroll(
            collection_name=COLLECTION, limit=1000, offset=offset, with_payload=False
        )
        existing.update(str(p.id) for p in pts)
        if offset is None:
            break
    print(f"resuming: {len(existing)} passages already indexed")

    buffer: list[dict] = []
    batch = 128
    with tqdm(total=len(selected), desc="textbooks") as bar:
        for i in selected:
            row = ds[i]
            content = (row["content"] or "").strip()
            if len(content) < 80:
                continue
            pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"tb:{row['id']}"))
            bar.update(1)
            if pid in existing:
                continue
            buffer.append(
                {
                    "id": pid,
                    "text": content,
                    "title": row["title"],
                    "url": f"https://huggingface.co/datasets/MedRAG/textbooks ({row['title']})",
                    "pmid": row["id"],
                    "source": row["title"],
                    "topic": "textbook",
                    "pubtypes": ["Textbook"],
                    "year": "",
                    "chunk_index": 0,
                }
            )
            if len(buffer) >= batch:
                _flush(client, buffer, PointStruct)
                buffer = []
        if buffer:
            _flush(client, buffer, PointStruct)

    print(f"collection '{COLLECTION}' size: {client.count(COLLECTION).count}")


def _flush(client, buffer: list[dict], PointStruct) -> None:
    vectors = embed([p["text"] for p in buffer])
    client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(id=p["id"], vector=v, payload={k: val for k, val in p.items() if k != "id"})
            for p, v in zip(buffer, vectors)
        ],
    )


if __name__ == "__main__":
    main()
