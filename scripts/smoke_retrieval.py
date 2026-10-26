"""Smoke test: does retrieval return relevant, cited passages for a MedQA question?"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.retrieval import search_with_fallback  # noqa: E402


def main() -> None:
    sample = config.SAMPLES_DIR / "medqa_sample.jsonl"
    rows = [json.loads(l) for l in sample.open(encoding="utf-8") if l.strip()]
    row = rows[0]
    print("QUESTION:", row["question"][:300])
    print("GOLD:", row["answer_idx"], row["answer_text"][:120])
    hits, trace = search_with_fallback(row["question"], simplified_query=" ".join(row["question"].split()[:12]))
    print("\nfallback trace:", json.dumps(trace["levels"], indent=1))
    for i, h in enumerate(hits, 1):
        print(f"\n[{i}] score={h['score']:.3f} pmid={h['pmid']} {h['title'][:90]}")
        print("   ", h["text"][:260].replace("\n", " "))
        print("   ", h["url"])


if __name__ == "__main__":
    main()
