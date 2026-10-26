"""AGENT 1 - Clinical Text Clarifier (paper II-A-1).

Paper: normalisation -> NER -> relation extraction -> SNOMED CT/ICD-10 mapping,
then zero-shot structuring into JSON. This implementation keeps the same output
contract but delegates entity/relation extraction to the LLM; ICD-10/SNOMED
coding is recorded as a known deviation (see README).
"""
from __future__ import annotations

from typing import Any

from .llm import generate_json

PROMPT = """You are the Clinical Text Clarifier of a clinical decision support system.
Convert the input into structured clinical information.

Return ONLY this JSON object:
{{
  "symptoms": ["..."],
  "duration": "text or null",
  "severity": "text or null",
  "medications": ["..."],
  "test_results": ["..."],
  "patient_age": "text or null",
  "patient_sex": "text or null",
  "key_concepts": ["the main clinical concepts needed to search the literature"],
  "missing_information": ["clinically relevant facts that are absent"],
  "retrieval_query": "a search-engine style query of 8-15 words built ONLY from the input"
}}

Rules:
- Use ONLY facts present in the input. Never invent a symptom, duration, drug or result.
- Write "not provided" facts into missing_information instead of guessing.
- Preserve explicit negatives (e.g. "no fever") by prefixing "no " in the symptom string.
- retrieval_query must be suitable for semantic search over biomedical abstracts.
- If the input is an exam-style question, key_concepts must still describe the clinical content.

INPUT:
{question}
"""


def clarify(question: str) -> dict[str, Any]:
    data = generate_json(PROMPT.format(question=question))
    defaults = {
        "symptoms": [],
        "duration": None,
        "severity": None,
        "medications": [],
        "test_results": [],
        "patient_age": None,
        "patient_sex": None,
        "key_concepts": [],
        "missing_information": [],
        "retrieval_query": question,
    }
    out = {**defaults, **(data or {})}
    if not str(out.get("retrieval_query") or "").strip():
        out["retrieval_query"] = " ".join(map(str, out.get("key_concepts") or [])) or question
    return out
