"""Print a saved execution log in a form that is readable during a demo.

  python scripts/show_case.py medqa_347
  python scripts/show_case.py medqa_107
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def clip(text: str, n: int = 260) -> str:
    return textwrap.shorten(" ".join(str(text or "").split()), width=n, placeholder=" ...")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/show_case.py <case_id>")
    path = ROOT / "results" / "execution_logs" / f"{sys.argv[1]}.json"
    if not path.exists():
        raise SystemExit(f"missing {path}; run main.py --case {sys.argv[1]} first")
    d = json.loads(path.read_text(encoding="utf-8"))

    print("=" * 80)
    print(f"CASE {d['case_id']}  mode={d['mode']}")
    print("=" * 80)
    print("\nQUESTION\n" + textwrap.fill(d["question"], width=100))
    if d.get("options"):
        print("\nOPTIONS")
        for k, v in d["options"].items():
            print(f"  {k}. {v}")

    print("\nCLARIFIER")
    for k, v in (d.get("clinical") or {}).items():
        print(f"  {k}: {v}")

    for it in d.get("iterations", []):
        print("\n" + "-" * 80)
        print(f"ITERATION {it['iteration']}  confidence={it['confidence']}")
        print(f"  query: {it.get('query')}")
        print(f"  components: {it.get('confidence_components')}")
        print(f"  hypotheses: {it.get('hypotheses')}")
        print(f"  stop: {it.get('stop_reason')}")

        if it.get("passages"):
            print("  RAG passages:")
            for i, p in enumerate(it["passages"], 1):
                src = p.get("pmid") or p.get("source") or ""
                print(f"    [R{i}] {clip(p.get('title',''), 85)}")
                print(f"         {src}  score={p.get('score'):.3f}  {clip(p.get('text',''), 180)}")

        if it.get("web_records"):
            print("  PubMed records:")
            for i, r in enumerate(it["web_records"], 1):
                print(f"    [W{i}] {clip(r.get('title',''), 85)}")
                print(
                    f"         PMID {r.get('pmid')}  level {r.get('evidence_level')} "
                    f"({r.get('evidence_label')})  {clip(r.get('abstract',''), 180)}"
                )

        fb = it.get("feedback")
        if fb:
            print("  OPTIMIZER")
            print(f"    reason: {clip(fb.get('reason',''), 400)}")
            print(f"    refined: {fb.get('refined_query')}")

    print("\n" + "=" * 80)
    print("FINAL")
    sel = d.get("selected", {})
    print(f"  answer: {sel.get('answer') or sel.get('answer_text')}")
    print(f"  rationale: {clip(sel.get('rationale',''), 400)}")
    print(f"  confidence={d.get('confidence')}  iterations={d.get('n_iterations')}  "
          f"latency={d.get('latency_s')}s")


if __name__ == "__main__":
    main()
