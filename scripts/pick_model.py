"""Probe which Gemini Flash-class model actually answers on this key, and how fast.

  python scripts/pick_model.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

CANDIDATES = [
    "gemini-3.8-flash",
    "gemini-3.8-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-3.8-pro",
    "gemini-2.5-flash",
]


def probe(model: str) -> tuple[str, float, str]:
    t = time.time()
    try:
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            params={"key": config.GEMINI_API_KEY},
            json={
                "contents": [{"parts": [{"text": "Reply with one word: OK"}]}],
                "generationConfig": {"temperature": 0, "maxOutputTokens": 24},
            },
            timeout=(10, 45),
        )
        dt = round(time.time() - t, 1)
        if r.status_code == 200:
            txt = r.json()["candidates"][0]["content"]["parts"][0].get("text", "").strip()
            return "OK", dt, txt[:40]
        return str(r.status_code), dt, r.text[:120].replace("\n", " ")
    except Exception as exc:  # noqa: BLE001
        return "TIMEOUT" if "Timeout" in type(exc).__name__ else type(exc).__name__, round(
            time.time() - t, 1
        ), ""


def main() -> None:
    if not config.GEMINI_API_KEY:
        raise SystemExit("GEMINI_API_KEY empty in .env")
    best = None
    for m in CANDIDATES:
        status, dt, note = probe(m)
        print(f"{m:<28} {status:<8} {dt:>6}s  {note}")
        if status == "OK" and (best is None or dt < best[1]):
            best = (m, dt)
    print()
    if best:
        print(f"fastest working model -> {best[0]} ({best[1]}s)")
        print(f"set GEMINI_MODEL={best[0]} in .env")
    else:
        print("no model answered; keep retrying - 503 'high demand' is transient")


if __name__ == "__main__":
    main()
