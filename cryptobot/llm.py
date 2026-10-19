"""Thin LLM clients for the offline coach: Claude (official SDK) or a local model via Ollama.

One method, `complete(system, user, schema) -> LLMResult`, returning JSON that matches the
schema. This module is never imported by the trading loop and knows nothing about brokers.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import requests

from cryptobot.config import Config

log = logging.getLogger(__name__)

# USD per million tokens (input, output). Unknown models fall back to the most expensive row.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}


class LLMError(Exception):
    pass


@dataclass
class LLMResult:
    data: dict[str, Any] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    refused: bool = False
    raw: str = ""


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = PRICES_PER_MTOK.get(model, (10.0, 50.0))
    return input_tokens / 1e6 * price_in + output_tokens / 1e6 * price_out


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMError(f"model did not return valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise LLMError("model returned JSON that is not an object")
    return data


class ClaudeLLM:
    """Claude through the Anthropic Messages API with structured (JSON schema) output."""

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        api_key: str | None = None,
        client=None,
        max_tokens: int = 8000,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        if client is None:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover
                raise LLMError("pip install anthropic to use coach.provider: claude") from e
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise LLMError("ANTHROPIC_API_KEY is not set in .env")
            client = anthropic.Anthropic(api_key=key, timeout=timeout, max_retries=1)
        self.client = client

    def complete(
        self, system: str, user: str, schema: dict[str, Any], effort: str | None = None
    ) -> LLMResult:
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
        if effort:
            output_config["effort"] = effort
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config=output_config,
        )
        usage = getattr(resp, "usage", None)
        tokens_in = int(getattr(usage, "input_tokens", 0) or 0)
        tokens_out = int(getattr(usage, "output_tokens", 0) or 0)
        result = LLMResult(
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            cost_usd=estimate_cost(self.model, tokens_in, tokens_out),
            model=self.model,
        )
        stop = getattr(resp, "stop_reason", None)
        if stop == "refusal":
            result.refused = True
            return result
        if stop == "max_tokens":
            raise LLMError("response was cut off at max_tokens; raise coach max_tokens")
        text = "".join(
            getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text"
        )
        result.raw = text
        result.data = _parse_json(text)
        return result


class OllamaLLM:
    """A local model through Ollama's HTTP API. Free; nothing leaves the machine."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        host: str | None = None,
        session: requests.Session | None = None,
        timeout: float = 600.0,
    ) -> None:
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout

    def complete(
        self, system: str, user: str, schema: dict[str, Any], effort: str | None = None
    ) -> LLMResult:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": schema,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        try:
            r = self.session.post(f"{self.host}/api/chat", json=body, timeout=self.timeout)
            r.raise_for_status()
        except requests.RequestException as e:
            raise LLMError(f"Ollama request failed ({self.host}): {e}") from e
        js = r.json()
        text = (js.get("message") or {}).get("content", "")
        return LLMResult(
            data=_parse_json(text),
            input_tokens=int(js.get("prompt_eval_count") or 0),
            output_tokens=int(js.get("eval_count") or 0),
            cost_usd=0.0,
            model=self.model,
            raw=text,
        )


def make_llm(cfg: Config, provider: str | None = None, component: str = "coach"):
    """Build the client for the coach (default) or the in-loop decider."""
    section = {"decider": cfg.decider, "guardian": cfg.guardian}.get(component, cfg.coach)
    provider = provider or section.provider
    if provider == "claude":
        timeout = getattr(section, "timeout_seconds", 120.0)
        return ClaudeLLM(
            model=section.model,
            timeout=timeout,
            max_tokens=2000 if component == "decider" else 8000,
        )
    if provider == "ollama":
        return OllamaLLM(
            model=section.ollama_model, timeout=getattr(section, "timeout_seconds", 600.0)
        )
    if provider == "none":
        return None
    raise LLMError(f"unknown {component} provider {provider!r}")
