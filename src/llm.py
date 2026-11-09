"""Single LLM entry point so every agent shares retry, caching and JSON handling.

Uses the REST API directly (not the SDK) so we control timeouts explicitly: the
configured Flash model intermittently returns 503 "high demand" or stalls, and the
SDK's default timeouts are far too long for a 20-question evaluation run.

Providers (replication Phase 2):
  * ``gemini`` (default) - Google Gemini generateContent REST API, with the
    historical model-fallback chain and retry/backoff behaviour preserved.
  * any other provider name - OpenAI-compatible ``POST {base}/chat/completions``
    (covers OpenAI, Groq, OpenRouter, Together, Mistral, DeepSeek, xAI,
    Cerebras, ...). Credentials come from ``<PROVIDER>_API_KEY`` / optional
    ``<PROVIDER>_BASE_URL`` in .env - never hardcoded.

Each agent resolves its provider/model from ``config.AGENT_CONFIG`` (per-agent
.env overrides, shared Gemini default). If an optional provider fails (missing
key, auth error, timeout, bad model) the call FALLS BACK to the Gemini chain and
the reason is recorded, so the system never requires more than the one Gemini
key.

Every call appends an entry to a thread-local trace (agent, provider, model
actually executed, cached?, fallback?) which ``pipeline.run_case`` stores in the
execution log as ``model_trace``.

Cache key format is unchanged (``model|temperature|json_mode|prompt``, provider
excluded) so all pre-existing cache entries still hit.
"""
from __future__ import annotations

import hashlib
import json
import os
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

# Known OpenAI-compatible endpoints so users only set <PROVIDER>_API_KEY.
# Anything else must also set <PROVIDER>_BASE_URL in .env.
_BASE_URL_DEFAULTS = {
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "together": "https://api.together.xyz/v1",
    "mistral": "https://api.mistral.ai/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "xai": "https://api.x.ai/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "perplexity": "https://api.perplexity.ai",
}


class ProviderError(RuntimeError):
    """A non-Gemini provider call failed.

    ``retryable`` marks transient failures (429/5xx/timeouts) worth one short
    retry; non-retryable failures (bad key, unknown model, no config) go straight
    to the Gemini fallback.
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


# --- Thread-local model-execution trace --------------------------------------
_TRACE = threading.local()


def reset_trace() -> None:
    _TRACE.entries = []


def current_trace() -> list[dict[str, Any]]:
    return list(getattr(_TRACE, "entries", []))


def take_trace() -> list[dict[str, Any]]:
    entries = current_trace()
    reset_trace()
    return entries


def _trace_add(**entry: Any) -> None:
    entries = getattr(_TRACE, "entries", None)
    if entries is None:
        entries = []
        _TRACE.entries = entries
    entries.append(entry)


def _cache_file(key: str):
    return config.CACHE_DIR / f"{key}.json"


def _redact(text: str) -> str:
    for secret in (config.GEMINI_API_KEY, os.getenv("GEMINI_API_KEY", "")):
        if secret:
            text = text.replace(secret, "[redacted]")
    return text


def _resolve(agent: str | None, model: str | None) -> tuple[str, str]:
    """Resolve (provider, model) for a call. Explicit ``model`` wins over the
    agent config but keeps the agent's provider."""
    cfg = config.AGENT_CONFIG.get(agent or "", {})
    provider = cfg.get("provider") or "gemini"
    if model:
        return provider, model
    resolved = cfg.get("model") or ""
    if not resolved and provider == "gemini":
        resolved = config.GEMINI_MODEL
    return provider, resolved  # custom provider without a model -> ProviderError -> Gemini fallback


def _model_list(model: str) -> list[str]:
    """Gemini chain: the default model gets the full historical fallback chain;
    an explicitly overridden Gemini model is tried first, then the chain."""
    return list(dict.fromkeys([model, *MODEL_FALLBACKS]))


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


def _generate_gemini(
    prompt: str,
    model: str,
    temperature: float,
    json_mode: bool,
    retries: int,
) -> tuple[str, str, list[str]]:
    """Call the Gemini API with model chain + retry/backoff.

    Returns (text, model_that_actually_ran, chain_notes). Raises on exhaustion.
    """
    if not config.GEMINI_API_KEY:
        raise RuntimeError(
            f"GEMINI_API_KEY missing. Put it in {config.ROOT / '.env'} as GEMINI_API_KEY=..."
        )

    models = _model_list(model)
    m_idx = 0
    notes: list[str] = []
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
                return _parse_candidates(r.json()), current, notes

            msg = r.text[:400]
            if r.status_code == 404 and m_idx + 1 < len(models):
                m_idx += 1
                notes.append(f"{current} unavailable (404), advanced to {models[m_idx]}")
                print(f"[warn] {current} unavailable, switching to {models[m_idx]}")
                continue
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

    safe_error = _redact(str(last_err or "unknown error"))
    raise RuntimeError(f"LLM call failed after {retries} attempts: {safe_error}") from None


def _provider_env(provider: str) -> tuple[str, str]:
    """Credentials for a provider: <PROVIDER>_API_KEY / <PROVIDER>_BASE_URL in .env."""
    prefix = provider.upper().replace("-", "_")
    api_key = (os.getenv(f"{prefix}_API_KEY") or "").strip()
    base = (os.getenv(f"{prefix}_BASE_URL") or "").strip() or _BASE_URL_DEFAULTS.get(provider, "")
    return api_key, base


def _call_chat_provider(
    provider: str,
    model: str,
    prompt: str,
    temperature: float,
    json_mode: bool,
) -> str:
    """One non-Gemini provider call via an OpenAI-compatible chat endpoint."""
    prefix = provider.upper().replace("-", "_")
    if not model:
        raise ProviderError(
            f"provider '{provider}' has no model configured (set <AGENT>_MODEL in .env)"
        )
    api_key, base = _provider_env(provider)
    if not base:
        raise ProviderError(
            f"no base URL for '{provider}': set {prefix}_BASE_URL in .env "
            f"(known providers: {', '.join(sorted(_BASE_URL_DEFAULTS))})"
        )
    if not api_key:
        raise ProviderError(f"missing {prefix}_API_KEY in .env")

    url = f"{base.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    for attempt in range(3):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=(15, 120))
        except requests.RequestException as exc:
            if attempt >= 1:
                raise ProviderError(
                    f"cannot reach {provider}: {_redact(str(exc))[:200]}", retryable=True
                ) from None
            time.sleep(2)
            continue

        if r.status_code == 200:
            try:
                data = r.json()
            except ValueError:
                raise ProviderError(f"{provider} returned non-JSON output", retryable=True) from None
            choices = data.get("choices") or []
            content = ((choices[0].get("message") or {}).get("content") if choices else "") or ""
            if isinstance(content, list):  # some compatible servers return parts
                content = "".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            text = str(content).strip()
            if not text:
                raise ProviderError(f"{provider} returned an empty response", retryable=True)
            return text

        msg = _redact(r.text[:300])
        if r.status_code == 400 and json_mode and "response_format" in body:
            body.pop("response_format", None)  # many compatible servers reject json mode
            continue
        if r.status_code in (401, 403):
            raise ProviderError(
                f"{provider} auth failed ({r.status_code}): {msg} - check {prefix}_API_KEY"
            )
        if r.status_code == 404:
            raise ProviderError(f"{provider} model '{model}' not found (404): {msg}")
        if r.status_code == 429 or r.status_code >= 500:
            if attempt >= 2:
                raise ProviderError(
                    f"{provider} error {r.status_code}: {msg}", retryable=True
                )
            time.sleep(2 * (attempt + 1))
            continue
        raise ProviderError(f"{provider} error {r.status_code}: {msg}")

    raise ProviderError(f"{provider} call failed after retries")


def generate(
    prompt: str,
    model: str | None = None,
    json_mode: bool = False,
    temperature: float | None = None,
    use_cache: bool = True,
    retries: int = 4,
    agent: str | None = None,
) -> str:
    """Run one LLM call for ``agent`` (a key of ``config.AGENT_CONFIG``).

    Resolution order: explicit ``model`` > agent's <AGENT>_MODEL > shared Gemini
    default. Non-Gemini providers are tried first and fall back to Gemini on any
    failure; the executed provider/model is recorded in the thread-local trace.
    """
    t0 = time.time()
    provider, resolved_model = _resolve(agent, model)
    req_provider, req_model = provider, resolved_model  # as configured, before any fallback
    temperature = config.LLM_TEMPERATURE if temperature is None else temperature

    # Cache key format unchanged (provider excluded) -> existing entries still hit.
    key = hashlib.sha256(
        f"{resolved_model}|{temperature}|{json_mode}|{prompt}".encode("utf-8")
    ).hexdigest()[:32]
    cf = _cache_file(key)
    if use_cache and cf.exists():
        _trace_add(
            agent=agent,
            requested_provider=provider,
            requested_model=resolved_model,
            provider=provider,
            model=resolved_model,
            cached=True,
            fallback=False,
            fallback_reason=None,
            status="ok",
            duration_ms=round((time.time() - t0) * 1000),
        )
        return json.loads(cf.read_text(encoding="utf-8"))["text"]

    fallback_reasons: list[str] = []
    if provider != "gemini":
        try:
            text = _call_chat_provider(provider, resolved_model, prompt, temperature, json_mode)
            if use_cache:
                cf.write_text(
                    json.dumps(
                        {
                            "provider": provider,
                            "model": resolved_model,
                            "prompt": prompt,
                            "text": text,
                        },
                        ensure_ascii=False,
                        indent=1,
                    ),
                    encoding="utf-8",
                )
            _trace_add(
                agent=agent,
                requested_provider=provider,
                requested_model=resolved_model,
                provider=provider,
                model=resolved_model,
                cached=False,
                fallback=False,
                fallback_reason=None,
                status="ok",
                duration_ms=round((time.time() - t0) * 1000),
            )
            return text
        except Exception as exc:  # noqa: BLE001 - any provider failure -> Gemini fallback
            fallback_reasons.append(_redact(str(exc))[:300])
            print(
                f"[warn] agent={agent or '-'} provider={provider} failed -> "
                f"falling back to gemini ({fallback_reasons[-1][:160]})"
            )
            provider = "gemini"
            resolved_model = config.GEMINI_MODEL
            # Recompute the cache key under the Gemini model so the fallback
            # response is cached where the shared default would look for it.
            key = hashlib.sha256(
                f"{resolved_model}|{temperature}|{json_mode}|{prompt}".encode("utf-8")
            ).hexdigest()[:32]
            cf = _cache_file(key)

    try:
        text, model_used, chain_notes = _generate_gemini(
            prompt, resolved_model, temperature, json_mode, retries
        )
    except Exception:
        _trace_add(
            agent=agent,
            requested_provider=req_provider,
            requested_model=req_model,
            provider="gemini",
            model=resolved_model,
            cached=False,
            fallback=bool(fallback_reasons),
            fallback_reason="; ".join(fallback_reasons) or None,
            status="error",
            duration_ms=round((time.time() - t0) * 1000),
        )
        raise

    fallback_reasons.extend(chain_notes)
    if use_cache:
        cf.write_text(
            json.dumps(
                {
                    "provider": provider,
                    "model": model_used,
                    "prompt": prompt,
                    "text": text,
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
    _trace_add(
        agent=agent,
        requested_provider=req_provider,
        requested_model=req_model,
        provider=provider,
        model=model_used,
        cached=False,
        fallback=bool(fallback_reasons),
        fallback_reason="; ".join(fallback_reasons) or None,
        status="ok",
        duration_ms=round((time.time() - t0) * 1000),
    )
    return text


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
