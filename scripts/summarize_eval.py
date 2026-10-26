"""Summarise the newest evaluation CSV: accuracy table + paired mode comparison.

  python scripts/summarize_eval.py                 # newest eval_*.csv
  python scripts/summarize_eval.py --csv results/eval_medqa_....csv
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def as_bool(v: str) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    if args.csv:
        path = Path(args.csv)
    else:
        files = sorted((ROOT / "results").glob("eval_*.csv"), key=lambda p: p.stat().st_mtime)
        if not files:
            raise SystemExit("no eval_*.csv in results/")
        path = files[-1]

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_mode[r["mode"]].append(r)

    print(f"source: {path.name}\n")
    print("| mode | n | correct | accuracy | errors | mean conf | mean iters | s/case |")
    print("|---|---|---|---|---|---|---|---|")
    for mode in ("baseline", "rag", "full"):
        rs = by_mode.get(mode)
        if not rs:
            continue
        n = len(rs)
        correct = sum(as_bool(r["correct"]) for r in rs)
        confs = [float(r["confidence"]) for r in rs if r["confidence"] not in ("", "None")]
        iters = [float(r["iterations"]) for r in rs if r["iterations"] not in ("", "None")]
        lat = [float(r["latency_s"]) for r in rs if r["latency_s"] not in ("", "None")]
        errs = sum(1 for r in rs if r["error"])
        conf_txt = f"{sum(confs) / len(confs):.3f}" if confs else "-"
        print(
            f"| {mode} | {n} | {correct} | {correct / n * 100:.1f}% | {errs} | "
            f"{conf_txt} | {sum(iters) / len(iters):.2f} | "
            f"{sum(lat) / len(lat):.1f} |"
        )

    # Paired comparison: same case_id in two modes.
    def index(mode: str) -> dict[str, dict]:
        return {r["case_id"]: r for r in by_mode.get(mode, [])}

    base, full = index("baseline"), index("full")
    shared = sorted(set(base) & set(full))
    if shared:
        cells = {"both": 0, "only_base": 0, "only_full": 0, "neither": 0}
        for cid in shared:
            b, f = as_bool(base[cid]["correct"]), as_bool(full[cid]["correct"])
            key = "both" if b and f else "only_base" if b else "only_full" if f else "neither"
            cells[key] += 1
        print(
            f"\nBaseline vs full (n={len(shared)}): both {cells['both']}, "
            f"only baseline {cells['only_base']}, only full {cells['only_full']}, "
            f"neither {cells['neither']}"
        )

    wrong = [r for r in rows if not as_bool(r["correct"])]
    if wrong:
        print("\nIncorrect / failed cases:")
        for r in sorted(wrong, key=lambda x: (x["mode"], x["case_id"])):
            note = f" ERROR {r['error'][:80]}" if r["error"] else ""
            print(
                f"  {r['mode']:<8} {r['case_id']:<14} gold={r['gold']:<28} "
                f"pred={r['predicted']}{note}"
            )


if __name__ == "__main__":
    main()
