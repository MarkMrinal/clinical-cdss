"""Readable end-to-end demo for the presentation (synthetic case, no patient data).

  python scripts/demo_case.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.pipeline import run_case  # noqa: E402

CASE = (
    "A 64-year-old man presents with 3 days of fever up to 38.9 C, productive cough with "
    "yellow sputum, and sharp right-sided chest pain that worsens on inspiration. He has "
    "a history of type 2 diabetes and smokes 20 cigarettes a day. On examination: "
    "respiratory rate 24/min, SpO2 91% on room air, crackles at the right lung base. "
    "He has no recent travel and no leg swelling."
)


def main() -> None:
    res = run_case(CASE, mode="full", case_id="demo_pneumonia")

    print("=" * 78)
    print("CLINICAL TEXT CLARIFIER")
    for k, v in res["clinical"].items():
        print(f"  {k}: {v}")

    print("\nITERATIONS")
    for it in res["iterations"]:
        print(
            f"  [{it['iteration']}] conf={it['confidence']} "
            f"components={json.dumps(it['confidence_components'])}"
        )
        print(f"      query: {it['query']}")
        print(f"      hypotheses: {it['hypotheses']}")
        print(f"      web queries: {it['web_queries']}")
        print(f"      web records: {it['n_web_records']}  stop={it['stop_reason']}")
        if it.get("feedback"):
            print(f"      requery reason: {it['feedback']['reason']}")
            print(f"      new query: {it['feedback']['refined_query']}")

    print("\nFINAL SELECTION")
    print(json.dumps(res["selected"], ensure_ascii=False, indent=1))
    print(
        f"\nconfidence={res['confidence']}  iterations={res['n_iterations']}  "
        f"latency={res['latency_s']}s  threshold tau={config.TAU}"
    )
    print(f"audit log -> {config.LOG_DIR / (res['case_id'] + '.json')}")


if __name__ == "__main__":
    main()
