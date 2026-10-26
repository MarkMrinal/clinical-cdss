"""Download MedQA / PubMedQA / MedBullets and freeze fixed samples (with IDs).

The paper evaluates 50 questions per dataset but never states which ones.
We draw a seeded sample and persist the IDs so every run is comparable.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

MEDQA_CANDIDATES = [
    ("bigbio/med_qa", "med_qa_en_source"),  # 5-choice, as the paper describes
    ("GBaker/MedQA-USMLE-4-options", None),
]
PUBMEDQA_CANDIDATES = [
    ("qiaojin/PubMedQA", "pqa_labeled"),
    ("bigbio/pubmed_qa", "pubmed_qa_labeled_fold0_source"),
]
MEDBULLETS_CANDIDATES = [
    ("LangAGI-Lab/medbullets", None),
    ("mkieffer/Medbullets", None),
]


def load_first(candidates):
    from datasets import get_dataset_config_names, load_dataset

    errors = []
    for name, cfg_name in candidates:
        try:
            ds = load_dataset(name, cfg_name) if cfg_name else load_dataset(name)
            print(f"[ok] {name} cfg={cfg_name} -> splits {list(ds.keys())}")
            return ds, name, cfg_name
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}/{cfg_name}: {exc}")
            print(f"[fail] {name} cfg={cfg_name}: {str(exc)[:200]}")
    try:
        print("available configs:", get_dataset_config_names(candidates[0][0]))
    except Exception:  # noqa: BLE001
        pass
    raise RuntimeError("No candidate loaded:\n" + "\n".join(errors))


def pick_split(ds):
    for name in ("test", "validation", "train"):
        if name in ds:
            return ds[name], name
    first = next(iter(ds))
    return ds[first], first


def norm_options(options) -> dict[str, str]:
    if isinstance(options, dict):
        return {str(k): str(v) for k, v in options.items()}
    out = {}
    if options and isinstance(options[0], dict):
        for d in options:
            key = str(d.get("key", "")).strip().upper()
            val = str(d.get("value", "")).strip()
            if key and val:
                out[key] = val
        return out
    for i, opt in enumerate(options):
        out[chr(65 + i)] = str(opt)
    return out


def options_from_row(r: dict) -> dict[str, str]:
    """Handles MedQA-style {'options': {...}} and MedBullets-style opa..ope columns."""
    if "options" in r:
        return norm_options(r["options"])
    out = {}
    for letter in "abcde":
        val = r.get(f"op{letter}")
        if val is not None and str(val).strip().lower() not in ("", "nan", "none"):
            out[letter.upper()] = str(val).strip()
    return out


def main() -> None:
    random.seed(config.SAMPLE_SEED)
    n = config.SAMPLE_N

    ds, name, cfg = load_first(MEDQA_CANDIDATES)
    split, split_name = pick_split(ds)
    idx = list(range(len(split)))
    random.shuffle(idx)
    idx = sorted(idx[:n])
    rows = []
    for i in idx:
        r = split[i]
        opts = options_from_row(r)
        rows.append(
            {
                "id": f"medqa_{i}",
                "dataset": name,
                "split": split_name,
                "question": r["question"],
                "options": opts,
                "answer_idx": str(r.get("answer_idx") or r.get("answer")),
                "answer_text": opts.get(str(r.get("answer_idx") or r.get("answer")), ""),
            }
        )
    p = config.SAMPLES_DIR / "medqa_sample.jsonl"
    p.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows), encoding="utf-8")
    print(f"[saved] {len(rows)} MedQA -> {p}")

    ds, name, cfg = load_first(PUBMEDQA_CANDIDATES)
    split, split_name = pick_split(ds)
    idx = list(range(len(split)))
    random.shuffle(idx)
    idx = sorted(idx[:n])
    rows = []
    for i in idx:
        r = split[i]
        ctx = r.get("context") or {}
        contexts = ctx.get("contexts") if isinstance(ctx, dict) else ctx
        rows.append(
            {
                "id": f"pubmedqa_{r.get('pubid', i)}",
                "dataset": name,
                "split": split_name,
                "question": r["question"],
                "contexts": contexts or [],
                "answer_text": r.get("long_answer", ""),
                "final_decision": r.get("final_decision", ""),
            }
        )
    p = config.SAMPLES_DIR / "pubmedqa_sample.jsonl"
    p.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows), encoding="utf-8")
    print(f"[saved] {len(rows)} PubMedQA -> {p}")

    ds, name, cfg = load_first(MEDBULLETS_CANDIDATES)
    split, split_name = pick_split(ds)
    idx = list(range(len(split)))
    random.shuffle(idx)
    idx = sorted(idx[:n])
    rows = []
    for i in idx:
        r = split[i]
        opts = options_from_row(r)
        rows.append(
            {
                "id": f"medbullets_{cfg or 'x'}_{i}",
                "dataset": name,
                "split": split_name,
                "question": r["question"],
                "options": opts,
                "answer_idx": str(r.get("answer_idx") or r.get("answer")),
                "answer_text": opts.get(str(r.get("answer_idx") or r.get("answer")), ""),
            }
        )
    p = config.SAMPLES_DIR / "medbullets_sample.jsonl"
    p.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows), encoding="utf-8")
    print(f"[saved] {len(rows)} MedBullets -> {p}")


if __name__ == "__main__":
    main()
