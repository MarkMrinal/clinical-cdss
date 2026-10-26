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

# --- Paper-reported settings -------------------------------------------------
EMBED_MODEL = os.getenv(
    "EMBED_MODEL", "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
)
COLLECTION = "biomedical"
COLLECTIONS = ["biomedical", "textbooks"]  # paper: textbooks + guidelines + literature
TAU = float(os.getenv("TAU", "0.65"))          # paper: confidence threshold
THETA = float(os.getenv("THETA", "0.80"))      # paper: free-text correctness threshold
MAX_ITERS = int(os.getenv("MAX_ITERS", "3"))    # paper: max optimisation cycles
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
TOP_K = int(os.getenv("TOP_K", "5"))
WEB_MAX_RECORDS = int(os.getenv("WEB_MAX_RECORDS", "4"))
# Let the scanner build PubMed queries with the LLM (paper: Qwen) or by rule.
# Set to 0 if the API is rate-limited: saves up to 3 calls per iteration.
SCANNER_LLM_QUERY = os.getenv("SCANNER_LLM_QUERY", "1") == "1"

# --- Models ------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
# NOTE: the paper used gemini-2.0-flash. It is retired on current keys (API returns
# "no longer available" and redirects to gemini-3.8-flash). Documented deviation.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

# --- Dev / experiment --------------------------------------------------------
SAMPLE_SEED = int(os.getenv("SAMPLE_SEED", "42"))
SAMPLE_N = int(os.getenv("SAMPLE_N", "50"))

for _d in (RAW_DIR, SAMPLES_DIR, RESULTS_DIR, LOG_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)
