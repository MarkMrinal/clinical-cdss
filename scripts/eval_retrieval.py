"""Retrieval-only diagnostic: does the top-k context ever contain the gold answer?

Cheap to run (no LLM) and directly comparable across corpus versions.
This is NOT the paper's accuracy metric - it is an upper bound on what the RAG
stream can contribute.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.retrieval import search_with_fallback  # noqa: E402

STOP = {"the", "of", "and", "with", "in", "a", "an", "to", "for", "or", "on", "by"}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", s.lower())


def key_terms(answer: str) -> set[str]:
    toks = [t for t in norm(answer).split() if len(t) > 2 and t not in STOP]
    return set(toks)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    rows = [
        json.loads(l)
        for l in (config.SAMPLES_DIR / "medqa_sample.jsonl").open(encoding="utf-8")
        if l.strip()
    ][:n]

    hits_any = 0
    hits_rank1 = 0
    scores = []
    textbook_share = []
    for r in rows:
        passages, _ = search_with_fallback(r["question"])
        terms = key_terms(r.get("answer_text", ""))
        found_at = None
        for i, p in enumerate(passages, 1):
            if terms and len(terms & set(norm(p["text"]).split())) / len(terms) >= 0.6:
                found_at = i
                break
        if found_at:
            hits_any += 1
            if found_at == 1:
                hits_rank1 += 1
        if passages:
            scores.append(max(p["score"] for p in passages))
            textbook_share.append(
                sum(1 for p in passages if "textbooks" in (p.get("collections") or []))
                / len(passages)
            )
        print(
            f"{r['id']:<12} gold={r.get('answer_idx')} "
            f"({'found@' + str(found_at) if found_at else 'absent'})  "
            f"top_score={max((p['score'] for p in passages), default=0):.3f}  "
            f"| {r.get('answer_text','')[:45]}"
        )

    print(f"\nquestions: {len(rows)}")
    print(f"gold answer appears in top-k : {hits_any}/{len(rows)} ({hits_any/len(rows)*100:.0f}%)")
    print(f"gold answer at rank 1       : {hits_rank1}/{len(rows)} ({hits_rank1/len(rows)*100:.0f}%)")
    if scores:
        print(f"mean top-1 similarity       : {sum(scores)/len(scores):.3f}")
    if textbook_share:
        print(f"textbook share of top-k     : {sum(textbook_share)/len(textbook_share)*100:.0f}%")


if __name__ == "__main__":
    main()
