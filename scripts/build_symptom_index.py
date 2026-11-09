"""Third vector collection: the paper's public Symptom-Disease Dataset (ref [30]).

Paper II-B-1 / Fig. 3: the corpus pipeline combines "Medical Articles and
Textbooks + Symptom Disease Dataset"; ref [30] is D. Tecblic's HuggingFace
dataset `dux-tecblic/symptom-disease-dataset` (7,043 patient-style narratives,
1,082 disease classes).

SAFETY: this script creates/refreshes ONLY the new collection `symptom_disease`.
It never touches `biomedical` or `textbooks` (scripts/build_index.py deletes its
own collection on rerun, so every new corpus source must use a new name).
If `symptom_disease` already contains points, the script refuses to run unless
you pass --recreate (which deletes only that collection).

Passage construction (our cleaning choice - the paper does not say how it turned
rows into passages): "<narrative> Condition: <disease name>" so each row becomes
a symptom -> condition association passage. Long rows are split with a
recursive-style 512/64 character splitter (paper Table 2 chunk parameters);
rows <= 60 characters are dropped as too-brief (same rule as build_index.py).

Usage:  python scripts/build_symptom_index.py [--recreate]
"""
from __future__ import annotations

import csv
import io
import json
import sys
import uuid
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.retrieval import _get_client, embed  # noqa: E402

BASE = "https://huggingface.co/datasets/dux-tecblic/symptom-disease-dataset/resolve/main/"
FILES = ["symptom-disease-train-dataset.csv", "symptom-disease-test-dataset.csv"]
MAPPING = "mapping.json"
COLLECTION = "symptom_disease"
DIM = 768
CACHE_DIR = config.RAW_DIR / "symptom_disease"
CHUNK_SIZE = config.CHUNK_SIZE      # paper Table 2: 512
CHUNK_OVERLAP = config.CHUNK_OVERLAP  # paper Table 2: 64
MIN_CHUNK = 60                      # same too-brief rule as build_index.py


def fetch(name: str) -> str:
    """Download once, then reuse the local copy (offline rebuilds possible)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / name
    if path.exists() and path.stat().st_size > 0:
        return path.read_text(encoding="utf-8")
    import requests

    r = requests.get(BASE + name, timeout=90)
    r.raise_for_status()
    path.write_text(r.text, encoding="utf-8")
    return r.text


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Recursive-style character splitter (sentence/space aware), 512/64."""
    text = " ".join(text.split())
    if len(text) <= size:
        return [text] if text else []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            window = text[start:end]
            cut = max(window.rfind(". "), window.rfind("; "), window.rfind(" "))
            if cut > size * 0.6:
                end = start + cut + 1
        chunk = text[start:end].strip()
        if len(chunk) >= MIN_CHUNK:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def load_rows() -> list[dict]:
    label_to_name = {int(v): k for k, v in json.loads(fetch(MAPPING)).items()}
    rows: list[dict] = []
    for fname in FILES:
        split = "train" if "train" in fname else "test"
        reader = csv.DictReader(io.StringIO(fetch(fname)))
        for i, r in enumerate(reader):
            text = (r.get("text") or "").strip()
            if not text:
                continue
            try:
                label = int(r.get("label"))
            except (TypeError, ValueError):
                continue
            rows.append(
                {
                    "text": text,
                    "label": label,
                    "disease": label_to_name.get(label, f"class_{label}"),
                    "split": split,
                    "row": i,
                }
            )
    return rows


def main() -> None:
    from qdrant_client.models import Distance, PointStruct, VectorParams

    client = _get_client()
    recreate = "--recreate" in sys.argv

    existing = 0
    try:
        existing = client.count(COLLECTION, exact=True).count
    except Exception:  # noqa: BLE001 - collection does not exist yet
        pass
    if existing:
        if not recreate:
            raise SystemExit(
                f"collection '{COLLECTION}' already has {existing} points; refusing to "
                "overwrite. Re-run with --recreate to rebuild ONLY this new collection "
                "(existing 'biomedical'/'textbooks' collections are never touched)."
            )
        print(f"recreating collection '{COLLECTION}' ({existing} old points removed)...")
        client.delete_collection(COLLECTION)
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            COLLECTION, vectors_config=VectorParams(size=DIM, distance=Distance.COSINE)
        )

    rows = load_rows()
    print(f"loaded {len(rows)} rows from {len(FILES)} CSVs")

    payloads: list[dict] = []
    texts: list[str] = []
    for row in tqdm(rows, desc="splitting"):
        passage = f"{row['text']} Condition: {row['disease']}"
        for ci, chunk in enumerate(split_text(passage)):
            texts.append(chunk)
            payloads.append(
                {
                    "text": chunk,
                    "length": len(chunk),
                    "title": row["disease"],
                    "source": "dux-tecblic/symptom-disease-dataset",
                    "kind": "symptom_disease",
                    "topic": row["disease"],
                    "disease": row["disease"],
                    "label": row["label"],
                    "split": row["split"],
                    "chunk_index": ci,
                    "row": row["row"],
                }
            )
    print(f"{len(texts)} chunks to embed")

    batch = 64
    points: list[PointStruct] = []
    for i in tqdm(range(0, len(texts), batch), desc="embedding"):
        chunk_texts = texts[i : i + batch]
        vecs = embed(chunk_texts)
        for j, (v, t) in enumerate(zip(vecs, chunk_texts)):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()), vector=v, payload=payloads[i + j]
                )
            )
        if len(points) >= 512:
            client.upsert(COLLECTION, points, wait=True)
            points = []
    if points:
        client.upsert(COLLECTION, points, wait=True)

    count = client.count(COLLECTION, exact=True).count
    print(f"DONE: collection '{COLLECTION}' now has {count} points "
          f"(existing collections untouched)")


if __name__ == "__main__":
    main()
