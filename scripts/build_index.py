"""Chunk PubMed abstracts, embed with BioBERT, upsert into Qdrant (paper Sec. II-B-1)."""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.retrieval import embed  # noqa: E402

SEPARATORS = ["\n\n", "\n", ". ", "; ", " "]


def split_text(text: str, size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            cut = -1
            for sep in SEPARATORS:
                pos = window.rfind(sep)
                if pos > size // 3:
                    cut = start + pos + len(sep)
                    break
            if cut > start:
                end = cut
        chunk = text[start:end].strip()
        if len(chunk) > 60:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def main() -> None:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    src_file = config.RAW_DIR / "pubmed_abstracts.jsonl"
    if not src_file.exists():
        raise SystemExit(f"Missing {src_file}. Run scripts/fetch_pubmed.py first.")

    records = [json.loads(line) for line in src_file.open(encoding="utf-8") if line.strip()]
    print(f"loaded {len(records)} abstracts")

    points: list[dict] = []
    for rec in records:
        body = f"{rec['title']}\n\n{rec['abstract']}"
        for i, chunk in enumerate(split_text(body, config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
            points.append(
                {
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{rec['pmid']}#{i}")),
                    "text": chunk,
                    "title": rec["title"],
                    "url": rec["url"],
                    "pmid": rec["pmid"],
                    "source": rec["journal"],
                    "topic": rec.get("topic", ""),
                    "pubtypes": rec.get("pubtypes", []),
                    "year": rec.get("year", ""),
                    "chunk_index": i,
                }
            )
    print(f"created {len(points)} chunks (size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")

    client = QdrantClient(path=str(config.INDEX_DIR))
    if client.collection_exists(config.COLLECTION):
        client.delete_collection(config.COLLECTION)
    client.create_collection(
        collection_name=config.COLLECTION,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

    buffer: list[dict] = []
    batch = 128
    with tqdm(total=len(points), desc="embed+upsert") as bar:
        for pt in points:
            buffer.append(pt)
            if len(buffer) >= batch:
                _flush(client, buffer, PointStruct)
                bar.update(len(buffer))
                buffer = []
        if buffer:
            _flush(client, buffer, PointStruct)
            bar.update(len(buffer))

    print(f"collection '{config.COLLECTION}' size: {client.count(config.COLLECTION).count}")


def _flush(client, buffer: list[dict], PointStruct) -> None:
    texts = [p["text"] for p in buffer]
    vectors = embed(texts)
    client.upsert(
        collection_name=config.COLLECTION,
        points=[
            PointStruct(id=p["id"], vector=v, payload={k: val for k, val in p.items() if k != "id"})
            for p, v in zip(buffer, vectors)
        ],
    )


if __name__ == "__main__":
    main()
