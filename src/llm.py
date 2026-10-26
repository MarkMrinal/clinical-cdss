"""Single LLM entry point so every agent shares retry, caching and JSON handling.

Uses the REST API directly (not the SDK) so we control timeouts explicitly: the
configured Flash model intermittently returns 503 "high demand" or stalls, and the
SDK's default timeouts are far too long for a 20-question evaluation run.
"""
from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from typing import Any

import requests

from . import config

MODEL_FALLBACKS = list(
    dict.fromkeys(
        [config.GEMINI_MODEL, "gemini-3.8-flash", "gemini-flash-latest", "gemini-2.5-flash"]
    )
)
_RETRYABLE = ("429", "500", "502", "503", "504", "RESOURCE_EXHAUSTED", "overloaded", "high demand")


def _cache_file(key: str):
    return config.CACHE_DIR / f"{key}.json"


def _model_list(model: str | None) -> list[str]:
    model = model or config.GEMINI_MODEL
    return MODEL_FALLBACKS if model == config.GEMINI_MODEL else [model]


def _parse_candidates(payload: dict) -> str:
    cands = payload.get("candidates") or []
    if not cands:
        fb = payload.get("promptFeedback", {})
        raise RuntimeError(f"no candidates returned: {fb}")
    parts = (cands[0].get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts).strip()
    if not text:
        raise RuntimeError("empty response text")
    return text


def generate(
    prompt: str,
    model: str | None = None,
    json_mode: bool = False,
    temperature: float | None = None,
    use_cache: bool = True,
    retries: int = 4,
) -> str:
    model = model or config.GEMINI_MODEL
    temperature = config.LLM_TEMPERATURE if temperature is None else temperature

    key = hashlib.sha256(
        f"{model}|{temperature}|{json_mode}|{prompt}".encode("utf-8")
    ).hexdigest()[:32]
    cf = _cache_file(key)
    if use_cache and cf.exists():
        return json.loads(cf.read_text(encoding="utf-8"))["text"]

    if not config.GEMINI_API_KEY:
        raise RuntimeError(
            f"GEMINI_API_KEY missing. Put it in {config.ROOT / '.env'} as GEMINI_API_KEY=..."
        )

    models = _model_list(model)
    m_idx = 0
    last_err: Exception | None = None

    for attempt in range(retries):
        current = models[m_idx]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{current}:generateContent"
        body: dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature},
        }
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"

        try:
            r = requests.post(
                url,
                headers={"x-goog-api-key": config.GEMINI_API_KEY},
                json=body,
                timeout=(15, 120),
            )
            if r.status_code == 200:
                text = _parse_candidates(r.json())
                if use_cache:
                    cf.write_text(
                        json.dumps(
                            {"model": current, "prompt": prompt, "text": text},
                            ensure_ascii=False,
                            indent=1,
                        ),
                        encoding="utf-8",
                    )
                return text

            msg = r.text[:400]
            if r.status_code == 404 and m_idx + 1 < len(models):
                m_idx += 1
                print(f"[warn] {current} unavailable, switching to {models[m_idx]}")
                continue
            if any(code in str(r.status_code) for code in ("429", "500", "502", "503", "504")):
                raise RuntimeError(f"{r.status_code} {msg}")
            raise RuntimeError(f"{r.status_code} {msg}")
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            quota = any(tok in str(exc) for tok in _RETRYABLE)
            if quota:
                wait = min(75, 8 * (attempt + 1)) + random.uniform(0, 4)
            else:
                wait = 2 * (attempt + 1)
            time.sleep(wait)

    if isinstance(last_err, requests.exceptions.ConnectionError):
        # requests includes the full URL in connection exceptions; this request
        # puts the API key in the query string, so never surface that exception.
        raise RuntimeError(
            "Could not connect to the Gemini API. Check internet access and whether "
            "your firewall or network policy allows outbound HTTPS."
        ) from None
    if isinstance(last_err, requests.exceptions.Timeout):
        raise RuntimeError(
            "The Gemini API request timed out. Check connectivity and try again."
        ) from None

    safe_error = str(last_err or "unknown error")
    safe_error = safe_error.replace(config.GEMINI_API_KEY, "[redacted]")
    raise RuntimeError(f"LLM call failed after {retries} attempts: {safe_error}") from None


def generate_json(prompt: str, **kwargs: Any) -> Any:
    raw = generate(prompt, json_mode=True, **kwargs)
    return parse_json(raw)


def parse_json(raw: str) -> Any:
    """Tolerant JSON parse: handles ```json fences and leading prose."""
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = s.find(opener)
        end = s.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(s[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Could not parse JSON from model output: {raw[:300]!r}")
