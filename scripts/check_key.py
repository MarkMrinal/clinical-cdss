"""Validate the Gemini key without ever printing it."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.llm import generate  # noqa: E402


def main() -> None:
    key = config.GEMINI_API_KEY
    if not key:
        print(f"[FAIL] GEMINI_API_KEY is empty in {config.ROOT / '.env'}")
        print("       Open .env in Notepad and set GEMINI_API_KEY=AIza...")
        raise SystemExit(1)
    print(f"key found: {key[:4]}...{key[-4:]} ({len(key)} chars)")
    print(f"model: {config.GEMINI_MODEL}")
    try:
        out = generate("Reply with exactly one word: OK", use_cache=False, retries=3)
        print(f"[OK] model replied: {out[:80]!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {type(exc).__name__}: {str(exc)[:500]}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
