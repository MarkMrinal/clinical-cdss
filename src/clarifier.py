"""AGENT 1 - Clinical Text Clarifier (paper II-A-1).

The paper's sequential stages (Sec. II-A-1), in order:
  1. Text normalisation   - noise, abbreviations, spelling        -> deterministic (below)
  2. NER                  - classes: Symptom/Disease/Medication/Test Result -> LLM zero-shot
  3. Association extraction - entity-to-entity semantic context   -> LLM zero-shot
  4. Terminology mapping  - SNOMED CT / ICD-10 codes              -> deterministic, NLM
                           ICD-10-CM API via src/coding.py (SNOMED deferred: licence)
  5. Structured output    - predefined JSON schema                -> same LLM pass as 2+3

Stages 2, 3 and 5 share ONE zero-shot LLM call (the paper never states how many
model calls its Clarifier makes); stages 1 and 4 are deterministic and run outside
the model. The original output contract is unchanged - new fields (normalized_text,
entities, associations, codes, unmapped_concepts, coding_status) are additive, so
every downstream agent keeps working.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

from . import coding
from .llm import generate_json

# --- Stage 1: deterministic normalisation ------------------------------------
# High-precision clinical abbreviations only (case-sensitive: uppercase matches
# uppercase). Anything ambiguous (CP, PT, OD, HA as "headache"? kept out) is left
# for the LLM stage instead of being guessed.
_ABBREV = {
    "HTN": "hypertension",
    "HLD": "hyperlipidemia",
    "T2DM": "type 2 diabetes mellitus",
    "DM2": "type 2 diabetes mellitus",
    "T1DM": "type 1 diabetes mellitus",
    "DM1": "type 1 diabetes mellitus",
    "CAD": "coronary artery disease",
    "CHF": "congestive heart failure",
    "COPD": "chronic obstructive pulmonary disease",
    "MI": "myocardial infarction",
    "PE": "pulmonary embolism",
    "DVT": "deep vein thrombosis",
    "UTI": "urinary tract infection",
    "AKI": "acute kidney injury",
    "CKD": "chronic kidney disease",
    "GERD": "gastroesophageal reflux disease",
    "IBS": "irritable bowel syndrome",
    "ARDS": "acute respiratory distress syndrome",
    "CVA": "cerebrovascular accident",
    "TIA": "transient ischemic attack",
    "AF": "atrial fibrillation",
    "ABX": "antibiotics",
    "SOB": "shortness of breath",
    "DOE": "dyspnea on exertion",
    "N/V": "nausea and vomiting",
    "ER": "emergency room",
    "ICU": "intensive care unit",
    "CXR": "chest X-ray",
    "ECG": "electrocardiogram",
    "EKG": "electrocardiogram",
    "WBC": "white blood cell count",
    "PMHx": "past medical history",
    "PMH": "past medical history",
    "Hx": "history",
    "Dx": "diagnosis",
    "Tx": "treatment",
    "Rx": "prescription",
    "SX": "symptoms",
    "TB": "tuberculosis",
    "HIV": "human immunodeficiency virus",
    "AIDS": "acquired immunodeficiency syndrome",
    "FUO": "fever of unknown origin",
    "s/p": "status post",
    "w/": "with",
    "w/o": "without",
    "BID": "twice daily",
    "TID": "three times daily",
    "QID": "four times daily",
    "QD": "once daily",
    "PRN": "as needed",
    "PO": "by mouth",
    "IV": "intravenous",
}


def normalize_text(text: str) -> str:
    """Stage 1: NFKC unicode normalisation, whitespace cleanup, abbreviation
    expansion with word boundaries (never expands inside a longer token)."""
    s = unicodedata.normalize("NFKC", str(text or ""))
    s = re.sub(r"\s+", " ", s).strip()
    for abbr, expanded in _ABBREV.items():
        s = re.sub(
            rf"(?<![A-Za-z0-9]){re.escape(abbr)}(?![A-Za-z0-9])", expanded, s
        )
    # age normalisation: "45y/o" -> "45 years old"; "y/o 45" -> "aged 45"
    s = re.sub(r"(\d+)\s*y/o", r"\1 years old", s)
    s = re.sub(r"\by/o\s*(\d+)", r"aged \1", s)
    return s


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
  "retrieval_query": "a search-engine style query of 8-15 words built ONLY from the input",
  "entities": [{{"text": "...", "type": "Symptom"}}],
  "associations": [{{"entity_a": "...", "relation": "...", "entity_b": "..."}}]
}}

Rules:
- Use ONLY facts present in the input. Never invent a symptom, duration, drug or result.
- Write "not provided" facts into missing_information instead of guessing.
- Preserve explicit negatives (e.g. "no fever") by prefixing "no " in the symptom string.
- retrieval_query must be suitable for semantic search over biomedical abstracts.
- If the input is an exam-style question, key_concepts must still describe the clinical content.
- entities: extract named clinical concepts using EXACTLY these four type values
  ("Symptom", "Disease", "Medication", "Test Result"). No other type values.
- associations: link each entity to its semantic context - e.g. a symptom with its
  anatomical location, severity or duration, or a medication with the condition it
  treats. entity_a/entity_b are text spans; entity_b may be an anatomical site that
  is not itself one of the four types. Every entity that has such context in the
  input must appear in at least one association.

INPUT:
{question}
"""

_ENTITY_TYPES = {"Symptom", "Disease", "Medication", "Test Result"}


_TYPE_ALIASES = {
    "symptom": "Symptom", "symptoms": "Symptom",
    "disease": "Disease", "diseases": "Disease",
    "condition": "Disease", "conditions": "Disease",
    "diagnosis": "Disease", "diagnoses": "Disease",
    "disorder": "Disease", "disorders": "Disease",
    "finding": "Disease", "findings": "Disease",
    "medication": "Medication", "medications": "Medication",
    "drug": "Medication", "drugs": "Medication",
    "medicine": "Medication", "medicines": "Medication",
    "test result": "Test Result", "test results": "Test Result",
    "test": "Test Result", "tests": "Test Result",
    "lab": "Test Result", "labs": "Test Result",
    "lab result": "Test Result", "lab results": "Test Result",
    "laboratory test": "Test Result", "imaging": "Test Result",
    "investigation": "Test Result",
}


def _norm_type(raw: Any) -> str:
    t = str(raw or "").strip().lower()
    return _TYPE_ALIASES.get(t, "Other")  # "Other" kept for transparency; coding skips it


def _clean_entities(raw: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for e in raw if isinstance(raw, list) else []:
        if isinstance(e, str):
            e = {"text": e, "type": "Symptom"}
        if not isinstance(e, dict):
            continue
        text = str(e.get("text") or "").strip()
        if not text:
            continue
        etype = _norm_type(e.get("type"))
        if (text.lower(), etype) in seen:
            continue
        seen.add((text.lower(), etype))
        out.append({"text": text, "type": etype})
    return out


def _clean_associations(raw: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for a in raw if isinstance(raw, list) else []:
        if not isinstance(a, dict):
            continue
        entity_a = str(a.get("entity_a") or "").strip()
        relation = str(a.get("relation") or "").strip()
        entity_b = str(a.get("entity_b") or "").strip()
        if entity_a and relation and entity_b:
            out.append({"entity_a": entity_a, "relation": relation, "entity_b": entity_b})
    return out


def clarify(question: str) -> dict[str, Any]:
    # Stage 1: deterministic normalisation feeds the model (paper: sequential stages)
    normalized = normalize_text(question)

    # Stages 2 + 3 + 5: one zero-shot LLM pass with explicit staged outputs
    data = generate_json(PROMPT.format(question=normalized), agent="clarifier")

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
        "normalized_text": normalized,
        "entities": [],
        "associations": [],
        "codes": [],
        "unmapped_concepts": [],
        "coding_status": "not run",
    }
    out = {**defaults, **(data or {})}
    if not str(out.get("retrieval_query") or "").strip():
        out["retrieval_query"] = " ".join(map(str, out.get("key_concepts") or [])) or question

    # Sanitise staged outputs (a model can return the wrong shape)
    out["symptoms"] = [str(s) for s in out.get("symptoms") or [] if str(s).strip()]
    out["key_concepts"] = [str(s) for s in out.get("key_concepts") or [] if str(s).strip()]
    out["entities"] = _clean_entities(out.get("entities"))
    out["associations"] = _clean_associations(out.get("associations"))
    out["normalized_text"] = normalized

    # Stage 4: deterministic ICD-10-CM terminology mapping (never claims SNOMED;
    # map_concepts itself handles ICD10_ENABLED=0 -> status "disabled")
    out["codes"], out["unmapped_concepts"], out["coding_status"] = coding.map_concepts(
        out["entities"]
    )
    return out
