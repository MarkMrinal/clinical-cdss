"""Evaluation harness.

Compares the paper's four configurations on the SAME fixed questions:
  baseline : no retrieval
  rag      : biomedical retrieval only
  web      : external evidence only (PubMed + NICE + Serper)
  full     : RAG + web + fusion + adaptive feedback

Scoring is exact match on the option letter (MedQA / MedBullets) or, for free-text
datasets, an explicitly labelled semantic-similarity rule (paper: theta = 0.80).

Every row records correctness, confidence, iterations, latency, the evidence that
was actually retrieved (passage/web-record counts and which sources answered) and
any error - never a fabricated value.
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


def _evidence_payload(res: dict, row: dict, mode: str) -> dict:
    """Compact per-question evidence record for the sidecar JSONL.

    Keeps what each question actually consumed - local passages, external
    records (title/url/source/level), the Fig. 5 fallback rungs used, the
    per-agent model trace and the confidence breakdown - so an evaluation row
    can always be traced back to its evidence.
    """
    its = res.get("iterations") or []
    passages = [p for i in its for p in i.get("passages", [])]
    if not passages:
        passages = res.get("passages") or []
    web_records = [
        {
            "title": str(r.get("title", ""))[:160],
            "url": str(r.get("url") or r.get("link") or ""),
            "source": r.get("source"),
            "evidence_level": r.get("evidence_level"),
            "condition": r.get("condition"),
            "year": r.get("year") or "",
        }
        for i in its
        for r in i.get("web_records", [])
    ]
    return {
        "case_id": row["id"],
        "mode": mode,
        "hypotheses": [h for i in its for h in i.get("hypotheses", [])],
        "n_passages": len(passages),
        "passages": [
            {
                "title": str(p.get("title", ""))[:160],
                "score": p.get("score"),
                "source": p.get("source") or p.get("collection"),
            }
            for p in passages
        ],
        "n_web_records": len(web_records),
        "web_records": web_records,
        "evidence_sources": sorted({w["source"] for w in web_records if w["source"]}),
        "retrieval_trace": [
            {"iteration": i.get("iteration"), **(i.get("retrieval_trace") or {})}
            for i in its
            if i.get("retrieval_trace")
        ],
        "stop_reasons": [i.get("stop_reason") for i in its],
        # the optimizer's feedback report per iteration (reason/focus/
        # query_action/refined_query) - needed to analyse the loop honestly
        "feedback": [i.get("feedback") for i in its if i.get("feedback")],
        "confidence": res.get("confidence"),
        "confidence_components": res.get("confidence_components"),
        "confidence_factors": res.get("confidence_factors"),
        "diagnoses": res.get("diagnoses"),
        "search_failures": [f for i in its for f in i.get("search_failures", [])],
        "model_trace": res.get("model_trace"),
        "latency_s": res.get("latency_s"),
    }


def run_one(row: dict, mode: str, evidence: bool = True) -> dict:
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
        # Evidence actually retrieved for this question (audit columns):
        # how many local passages / external records were consumed and which
        # external sources answered. Absent streams report 0 / "" - never a guess.
        its = res.get("iterations") or []
        n_passages = sum(int(i.get("n_passages") or 0) for i in its)
        if not n_passages:
            n_passages = len(res.get("passages") or [])
        n_web = sum(int(i.get("n_web_records") or 0) for i in its)
        sources = sorted(
            {
                str(r.get("source"))
                for i in its
                for r in i.get("web_records", [])
                if r.get("source")
            }
        )
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
            "n_passages": n_passages,
            "n_web_records": n_web,
            "evidence_sources": "+".join(sources),
            "error": "",
            # sidecar payload (popped before the CSV is written): what evidence
            # this question actually consumed, so nothing is unauditable
            "_evidence": _evidence_payload(res, row, mode) if evidence else None,
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
            "n_passages": 0,
            "n_web_records": 0,
            "evidence_sources": "",
            "error": str(exc)[:300],
            "_evidence": None,
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="medqa")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument(
        "--modes",
        default="baseline,rag,web,full",
        help="comma list drawn from: baseline,rag,web,full (default: all four)",
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--evidence",
        default="1",
        help="write the per-question evidence sidecar JSONL (1/0, default 1)",
    )
    args = ap.parse_args()
    want_evidence = str(args.evidence).strip().lower() not in ("0", "false", "no")

    rows = load(args.dataset)[: args.n]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    print(f"{len(rows)} questions x {len(modes)} modes: {modes}")

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one, row, mode, want_evidence): (row["id"], mode)
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
    # Every NEW run is prefixed repl_ so replication results can never be mixed
    # up with the frozen prototype benchmark (eval_medqa_20261026_*).
    out_csv = config.RESULTS_DIR / f"repl_eval_{args.dataset}_{stamp}.csv"

    # Sidecar first (it still holds _evidence), then strip it for the CSV.
    sidecar = None
    if want_evidence:
        sidecar = config.RESULTS_DIR / f"repl_eval_{args.dataset}_{stamp}_evidence.jsonl"
        with sidecar.open("w", encoding="utf-8") as fh:
            for r in sorted(results, key=lambda r: (r["mode"], r["case_id"])):
                if r.get("_evidence"):
                    fh.write(json.dumps(r["_evidence"], ensure_ascii=False) + "\n")
    for r in results:
        r.pop("_evidence", None)

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

    out_json = config.RESULTS_DIR / f"repl_eval_{args.dataset}_{stamp}.json"
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
    if sidecar:
        print(f"EVIDENCE -> {sidecar}")


if __name__ == "__main__":
    main()
