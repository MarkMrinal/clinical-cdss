"""CLI entry point.

  python main.py --question "A 45-year-old man presents with fever, cough and fatigue..."
  python main.py --case medqa_12
  python main.py --question "..." --mode rag
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config  # noqa: E402
from src.pipeline import run_case  # noqa: E402


def load_sample(case_id: str) -> dict:
    for f in config.SAMPLES_DIR.glob("*.jsonl"):
        for line in f.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] == case_id:
                return row
    raise SystemExit(f"case {case_id!r} not found in {config.SAMPLES_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--question")
    ap.add_argument("--case")
    ap.add_argument("--mode", default="full", choices=["baseline", "rag", "full"])
    args = ap.parse_args()

    if args.case:
        row = load_sample(args.case)
        res = run_case(
            row["question"], row.get("options"), mode=args.mode, case_id=args.case
        )
        print("GOLD:", row.get("answer_idx"), row.get("answer_text"))
    elif args.question:
        res = run_case(args.question, None, mode=args.mode)
    else:
        raise SystemExit("provide --question or --case")

    print("\n=== SELECTED ===")
    print(json.dumps(res["selected"], ensure_ascii=False, indent=1))
    print(f"\nconfidence={res.get('confidence')} iterations={res.get('n_iterations')} "
          f"latency={res.get('latency_s')}s")
    for it in res.get("iterations", []):
        print(
            f"  iter {it['iteration']}: conf={it['confidence']} "
            f"rag={it['n_passages']} web={it['n_web_records']} stop={it['stop_reason']}"
        )
    print(f"\nlog -> {config.LOG_DIR}")


if __name__ == "__main__":
    main()
