"""Fetch PubMed abstracts via NCBI E-utilities (free, no API key needed).

Stand-in for the paper's Serper/PDF corpus: we index abstracts directly instead of
downloading 1,000 PDFs. Documented as a deviation in README.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, pubmed  # noqa: E402

OUT = config.RAW_DIR / "pubmed_abstracts.jsonl"

TOPICS = [
    "fever cough fatigue differential diagnosis",
    "chest pain dyspnea acute coronary syndrome",
    "abdominal pain differential diagnosis",
    "headache red flags",
    "community acquired pneumonia diagnosis",
    "diabetes mellitus diagnosis management",
    "hypertension diagnosis management",
    "acute kidney injury",
    "sepsis diagnosis management",
    "stroke diagnosis thrombolysis",
    "myocardial infarction",
    "asthma diagnosis management",
    "chronic obstructive pulmonary disease exacerbation",
    "thyroid dysfunction diagnosis",
    "anemia diagnosis",
    "meningitis diagnosis",
    "urinary tract infection diagnosis",
    "gastroesophageal reflux disease",
    "rheumatoid arthritis diagnosis",
    "major depressive disorder diagnosis",
    "pulmonary embolism diagnosis",
    "deep vein thrombosis",
    "pancreatitis diagnosis",
    "cirrhosis complications",
    "electrolyte disorders hyponatremia",
    "syncope evaluation",
    "altered mental status delirium",
    "back pain red flags",
    "skin infection cellulitis",
    "adverse drug reaction",
]


def main() -> None:
    seen: set[str] = set()
    n = 0
    with OUT.open("w", encoding="utf-8") as fh:
        for topic in tqdm(TOPICS, desc="topics"):
            try:
                ids = pubmed.esearch(f"{topic} AND hasabstract[text]", retmax=100)
                time.sleep(0.3)
                records = pubmed.efetch(ids)
                time.sleep(0.3)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] {topic}: {exc}")
                continue
            for rec in records:
                if rec["pmid"] in seen:
                    continue
                seen.add(rec["pmid"])
                rec["topic"] = topic
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
            print(f"  {topic}: {len(records)} records (total {n})")
    print(f"\nSaved {n} abstracts -> {OUT}")


if __name__ == "__main__":
    main()
