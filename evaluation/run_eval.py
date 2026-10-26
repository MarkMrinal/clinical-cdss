"""Evaluation harness.

Compares the paper's three configurations on the SAME fixed questions:
  baseline : no retrieval
  rag      : biomedical retrieval only
  full     : RAG + web + fusion + adaptive feedback

Scoring is exact match on the option letter (MedQA / MedBullets) or, for free-text
datasets, an explicitly labelled semantic-similarity rule (paper: theta = 0.80).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.pipeline import run_case  # noqa: E402


def load(dataset: str) -> list[dict]:
    path = config.SAMPLES_DIR / f"{dataset}_sample.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path}; run scripts/sample_datasets.py")
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def _cos(a: str, b: str) -> float:
    from src.retrieval import embed

    va, vb = embed([a, b])
    return sum(x * y for x, y in zip(va, vb))  # vectors are L2-normalised


def run_one(row: dict, mode: str) -> dict:
    t0 = time.time()
    try:
        res = run_case(
            row["question"],
            row.get("options"),
            mode=mode,
            case_id=f"{mode}__{row['id']}",
            save_log=False,
        )
        if row.get("options"):
            pred = res["selected"]["answer"]
            gold = row.get("answer_idx", "")
            correct = bool(pred) and str(pred).strip().upper() == str(gold).strip().upper()
        else:
            # Paper III: free-text correctness via sentence-embedding cosine >= theta.
            pred = res["selected"]["answer_text"]
            gold = row.get("answer_text", "")
            correct = bool(pred and gold) and _cos(pred, gold) >= config.THETA
        return {
            "case_id": row["id"],
            "mode": mode,
            "model": config.GEMINI_MODEL,
            "gold": gold if len(str(gold)) < 40 else str(gold)[:37] + "...",
            "predicted": pred,
            "correct": correct,
            "confidence": res.get("confidence"),
            "iterations": res.get("n_iterations"),
            "latency_s": round(time.time() - t0, 2),
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001 - record, never fabricate
        return {
            "case_id": row["id"],
            "mode": mode,
            "model": config.GEMINI_MODEL,
            "gold": row.get("answer_idx") or row.get("final_decision") or "",
            "predicted": "",
            "correct": False,
            "confidence": None,
            "iterations": 0,
            "latency_s": round(time.time() - t0, 2),
            "error": str(exc)[:300],
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="medqa")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--modes", default="baseline,rag,full")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    rows = load(args.dataset)[: args.n]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    print(f"{len(rows)} questions x {len(modes)} modes: {modes}")

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one, row, mode): (row["id"], mode)
            for mode in modes
            for row in rows
        }
        done = 0
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            done += 1
            flag = "OK " if r["correct"] else "ERR" if r["error"] else "X  "
            print(
                f"[{done}/{len(futures)}] {flag} {r['mode']:<8} {r['case_id']:<16} "
                f"gold={r['gold']} pred={r['predicted']} conf={r['confidence']} "
                f"it={r['iterations']} {r['latency_s']}s"
            )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = config.RESULTS_DIR / f"eval_{args.dataset}_{stamp}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(sorted(results, key=lambda r: (r["mode"], r["case_id"])))

    summary = {}
    for mode in modes:
        rs = [r for r in results if r["mode"] == mode]
        if not rs:
            continue
        n = len(rs)
        correct = sum(r["correct"] for r in rs)
        confs = [r["confidence"] for r in rs if r["confidence"] is not None]
        summary[mode] = {
            "n": n,
            "correct": correct,
            "accuracy": round(correct / n, 4),
            "errors": sum(1 for r in rs if r["error"]),
            "mean_confidence": round(sum(confs) / len(confs), 4) if confs else None,
            "mean_iterations": round(sum(r["iterations"] or 0 for r in rs) / n, 2),
            "mean_latency_s": round(sum(r["latency_s"] for r in rs) / n, 1),
        }

    out_json = config.RESULTS_DIR / f"eval_{args.dataset}_{stamp}.json"
    sample_path = config.SAMPLES_DIR / f"{args.dataset}_sample.jsonl"
    run_config = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_file": str(sample_path.relative_to(config.ROOT)),
        "dataset_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
        "n": len(rows),
        "modes": modes,
        "workers": args.workers,
        "model": config.GEMINI_MODEL,
        "temperature": config.LLM_TEMPERATURE,
        "embed_model": config.EMBED_MODEL,
        "tau": config.TAU,
        "theta": config.THETA,
        "max_iters": config.MAX_ITERS,
        "top_k": config.TOP_K,
        "scanner_llm_query": config.SCANNER_LLM_QUERY,
        "python": platform.python_version(),
    }
    out_json.write_text(
        json.dumps(
            {"config": vars(args), "run_config": run_config, "summary": summary},
            indent=1,
        ),
        encoding="utf-8",
    )

    print("\n=== SUMMARY ===")
    for mode, s in summary.items():
        print(
            f"{mode:<9} acc={s['accuracy']*100:5.1f}%  ({s['correct']}/{s['n']})  "
            f"errors={s['errors']}  conf={s['mean_confidence']}  iters={s['mean_iterations']}  "
            f"{s['mean_latency_s']}s/case"
        )
    print(f"\nCSV  -> {out_csv}\nJSON -> {out_json}")


if __name__ == "__main__":
    main()
