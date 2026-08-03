"""
LLM provider access — the only module that talks to Groq or OpenRouter.

Two providers, reached through two clients:
  - Groq       via the `groq` SDK          (router, re-ranker)
  - OpenRouter via the `openai` SDK        (synthesis, verifier)

OpenRouter speaks the OpenAI wire protocol, so the `openai` client works
against it by pointing base_url at config.OPENROUTER_BASE_URL.

Everything above this layer calls chat_text() or chat_json() and never
sees a provider-specific object. That keeps the agents testable with a
single monkeypatch and makes swapping a provider a one-file change.
"""

from __future__ import annotations

import json
import os
import re
import time
from functools import lru_cache

from src import config


class ProviderError(RuntimeError):
    """Raised when a provider call fails after all retries, or a key is missing.

    app.py catches this to show a graceful error instead of a stack trace.
    """


# --- Secrets ---------------------------------------------------------------

def _load_dotenv_once() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - python-dotenv is a declared dep
        return
    load_dotenv(override=False)


def get_secret(name: str) -> str:
    """Look up a secret from Streamlit secrets, then the environment, then .env.

    Streamlit is checked first so a Community Cloud deployment works with no
    .env file at all; os.environ is checked before .env so an explicitly
    exported value wins over a stale file.
    """
    try:
        import streamlit as st

        # st.secrets raises if no secrets.toml exists, which is the normal
        # case for local CLI runs (ingestion, tests) — fall through quietly.
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass

    value = os.getenv(name)
    if value:
        return value

    _load_dotenv_once()
    value = os.getenv(name)
    if value:
        return value

    raise ProviderError(
        f"{name} is not set. Copy .streamlit/secrets.toml.example to "
        f".streamlit/secrets.toml and fill it in, or export {name} in your "
        f"environment."
    )


# --- Clients ---------------------------------------------------------------

@lru_cache(maxsize=1)
def get_groq_client():
    from groq import Groq

    return Groq(
        api_key=get_secret("GROQ_API_KEY"),
        timeout=config.REQUEST_TIMEOUT_SECONDS,
    )


@lru_cache(maxsize=1)
def get_openrouter_client():
    from openai import OpenAI

    return OpenAI(
        api_key=get_secret("OPENROUTER_API_KEY"),
        base_url=config.OPENROUTER_BASE_URL,
        timeout=config.REQUEST_TIMEOUT_SECONDS,
    )


def _client_for(provider: str):
    if provider == "groq":
        return get_groq_client()
    if provider == "openrouter":
        return get_openrouter_client()
    raise ValueError(f"Unknown provider: {provider!r}")


# --- Calls -----------------------------------------------------------------

def chat_text(
    provider: str,
    model: str,
    system: str,
    user: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Single-turn chat completion returning raw text.

    Retries on transient failures with a short backoff, then raises
    ProviderError rather than letting a provider-specific exception escape.
    """
    client = _client_for(provider)
    last_error: Exception | None = None

    for attempt in range(config.MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=config.TEMPERATURE if temperature is None else temperature,
                max_tokens=max_tokens,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:  # provider SDKs raise their own hierarchies
            last_error = exc
            if attempt < config.MAX_RETRIES:
                time.sleep(2**attempt)

    raise ProviderError(
        f"{provider}/{model} failed after {config.MAX_RETRIES + 1} attempts: {last_error}"
    ) from last_error


def chat_json(
    provider: str,
    model: str,
    system: str,
    user: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dict:
    """Chat completion constrained to a JSON object, parsed into a dict.

    Asks the provider for JSON mode where supported and still parses
    defensively — a model that wraps its JSON in a ```json fence would
    otherwise take down the whole graph.
    """
    client = _client_for(provider)
    last_error: Exception | None = None

    for attempt in range(config.MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=config.TEMPERATURE if temperature is None else temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            return _parse_json(response.choices[0].message.content or "")
        except Exception as exc:
            last_error = exc
            if attempt < config.MAX_RETRIES:
                time.sleep(2**attempt)

    raise ProviderError(
        f"{provider}/{model} failed after {config.MAX_RETRIES + 1} attempts: {last_error}"
    ) from last_error


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _parse_json(raw: str) -> dict:
    """Parse a model's JSON reply, tolerating code fences and leading prose."""
    text = raw.strip()
    if not text:
        raise ValueError("empty response")

    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed
