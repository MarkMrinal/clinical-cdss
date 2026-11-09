"""Reproduce the Qdrant concurrency-fix evidence.

Runs 16 concurrent retrieval calls across 8 threads against the LOCAL index and
prints the collection counts, so the double-checked-lock fix in src/retrieval.py
can be re-verified after any change:

    python scripts/stress_qdrant.py

Expected: 16/16 ok and counts 11244 13691.
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

# Always test the local index, regardless of .env (cloud mode cannot be stress-tested
# from here and was never the backend of the reported evaluation).
config.QDRANT_URL = ""
config.QDRANT_API_KEY = ""

from src import retrieval  # noqa: E402


def task(i: int) -> tuple[str, int, str]:
    try:
        return ("ok", i, str(len(retrieval._raw_search([0.01] * 768, 5))))
    except Exception as exc:  # noqa: BLE001
        return ("err", i, f"{type(exc).__name__}: {exc}"[:120])


def main() -> None:
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(task, range(16)))
    ok = [r for r in results if r[0] == "ok"]
    errs = [r for r in results if r[0] == "err"]
    print(f"stress: {len(ok)}/16 ok, sample hits={ok[0][2] if ok else 'n/a'}")
    for e in errs:
        print(f"  ERROR thread {e[1]}: {e[2]}")

    client = retrieval._get_client()
    counts = {n: client.count(n).count for n in config.COLLECTIONS}
    print("counts:", " ".join(f"{k}={v}" for k, v in counts.items()))
    print("RESULT:", "PASS" if not errs and all(v > 0 for v in counts.values()) else "FAIL")


if __name__ == "__main__":
    main()
