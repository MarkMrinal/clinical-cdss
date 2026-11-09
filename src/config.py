"""Central configuration. Values mirror the base paper where stated, overridable via .env."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SAMPLES_DIR = DATA_DIR / "samples"
RESULTS_DIR = ROOT / "results"
LOG_DIR = RESULTS_DIR / "execution_logs"
CACHE_DIR = RESULTS_DIR / "llm_cache"
INDEX_DIR = RAW_DIR / "qdrant_index"
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()

# --- Paper-reported settings -------------------------------------------------
EMBED_MODEL = os.getenv(
    "EMBED_MODEL", "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
)
COLLECTION = "biomedical"
# paper: textbooks + guidelines + literature; Phase 4 adds the public
# Symptom-Disease Dataset (paper ref [30]) as its own collection - new sources
# always get a NEW collection name, existing ones are never rewritten.
COLLECTIONS = ["biomedical", "textbooks", "symptom_disease"]
TAU = float(os.getenv("TAU", "0.65"))          # paper: confidence threshold
THETA = float(os.getenv("THETA", "0.80"))      # paper: free-text correctness threshold
MAX_ITERS = int(os.getenv("MAX_ITERS", "3"))    # paper: max optimisation cycles
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
TOP_K = int(os.getenv("TOP_K", "5"))
WEB_MAX_RECORDS = int(os.getenv("WEB_MAX_RECORDS", "4"))

# --- Retrieval (replication Phase 4, paper Table 2 / Fig. 5) ------------------
# tau for EVIDENCE selection (distinct from TAU, which is the confidence gate).
RETRIEVAL_THRESHOLD = float(os.getenv("RETRIEVAL_THRESHOLD", "0.65"))
# Table 2 reports "term boosting: Active". The algorithm is unpublished, so
# _boost_query() implements a documented approximation (see src/retrieval.py).
TERM_BOOST = os.getenv("TERM_BOOST", "1") == "1"
# Let the scanner build PubMed queries with the LLM (paper: Qwen) or by rule.
# Set to 0 if the API is rate-limited: saves up to 3 calls per iteration.
SCANNER_LLM_QUERY = os.getenv("SCANNER_LLM_QUERY", "1") == "1"

# --- Phase 5: external web evidence (paper Sec. II-A-3) -----------------------
# The Serper key lives ONLY in .env (never committed, never logged, never echoed).
# With no key the scanner still runs on PubMed + NICE (both keyless) and records
# why Serper was skipped - one GEMINI_API_KEY is always enough to run the system.
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "").strip()
SERPER_ENABLED = os.getenv("SERPER_ENABLED", "1") == "1" and bool(SERPER_API_KEY)
# NICE guideline search runs directly against nice.org.uk (no key, no credits).
NICE_ENABLED = os.getenv("NICE_ENABLED", "1") == "1"
# One LLM pass turns the returned web pages into per-hypothesis summaries
# (paper: the scanner "derives text summaries pertinent to each hypothesis").
SCANNER_WEB_SUMMARY = os.getenv("SCANNER_WEB_SUMMARY", "1") == "1"
SERPER_CACHE_DIR = RESULTS_DIR / "serper_cache"  # disk cache: reproducible + no credit burn

# --- Models ------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
# NOTE: the paper used gemini-2.0-flash. It is retired on current keys (API returns
# "no longer available" and redirects to gemini-3.8-flash). Documented deviation.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

# --- Clinical text Clarifier (replication Phase 3) ----------------------------
# Stage 4 terminology mapping: free NLM ICD-10-CM API (no key). SNOMED CT is
# deferred (licence) and is never claimed. Set ICD10_ENABLED=0 to skip the API.
ICD10_ENABLED = os.getenv("ICD10_ENABLED", "1") == "1"
ICD10_MAX_QUERIES = int(os.getenv("ICD10_MAX_QUERIES", "8"))

# --- Dev / experiment --------------------------------------------------------
SAMPLE_SEED = int(os.getenv("SAMPLE_SEED", "42"))
SAMPLE_N = int(os.getenv("SAMPLE_N", "50"))

# --- Per-agent provider/model configuration (replication Phase 2) -------------
# The paper assigned four different LLMs to its five agents (DeepSeek-R1,
# Gemini-2.0-Flash, Qwen2-72B, GPT-4o-mini). We initially run ONE shared Gemini
# model for all five genuinely distinct agent roles (documented deviation), but
# every agent can be pointed at a different provider/model independently:
#
#   <AGENT>_PROVIDER = openai   (default: gemini, or LLM_PROVIDER if set)
#   <AGENT>_MODEL    = gpt-4o-mini
#
# where <AGENT> is one of: CLARIFIER, RAG_ANALYZER, EVIDENCE_SCANNER, FUSION,
# OPTIMIZER, SELECT (SELECT is the internal final-answer stage, not a paper agent).
#
# API keys NEVER live here: each provider reads "<PROVIDER>_API_KEY" from .env
# (e.g. GROQ_API_KEY=...). No key is required unless that agent is overridden,
# and a failing optional provider falls back to Gemini automatically (src/llm.py).
DEFAULT_PROVIDER = (os.getenv("LLM_PROVIDER", "") or "gemini").strip().lower()

AGENTS = ("clarifier", "rag_analyzer", "evidence_scanner", "fusion", "optimizer", "select")


def _agent_setting(agent: str) -> dict:
    prefix = agent.upper()
    provider = (os.getenv(f"{prefix}_PROVIDER", "") or DEFAULT_PROVIDER).strip().lower()
    model = (os.getenv(f"{prefix}_MODEL", "") or "").strip()
    if not model and provider == "gemini":
        model = GEMINI_MODEL  # shared default Gemini model
    return {"provider": provider, "model": model}


AGENT_CONFIG: dict[str, dict] = {a: _agent_setting(a) for a in AGENTS}

for _d in (RAW_DIR, SAMPLES_DIR, RESULTS_DIR, LOG_DIR, CACHE_DIR, SERPER_CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)
