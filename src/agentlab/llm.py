"""Chat-model clients.

Everything talks to models through one small interface, :class:`ChatModel`, using the
OpenAI chat-completions message format. That format is spoken by OpenAI, Groq, Gemini
(OpenAI-compatible endpoint), vLLM and Ollama, so one client covers hosted and local models.

* :class:`OpenAICompatibleModel` - real models over HTTP.
* :class:`ExtractiveModel` - a deterministic, LLM-free baseline: it answers with the most
  relevant retrieved passage. Used as a lower bound in evaluations and to run the whole
  pipeline in tests and CI without a model or API key.
* :class:`ScriptedModel` - replays pre-written responses; used in unit tests.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import requests

PROVIDERS = {
    "ollama": ("http://localhost:11434/v1", None),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
}


@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:8]}")
    raw_arguments: str = ""


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


class ChatModel(Protocol):
    name: str

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse: ...


_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def parse_text_tool_call(content: str, tool_names: set[str]) -> ToolCall | None:
    """Recover a tool call that a small model wrote as JSON text instead of using the
    native tool-calling field, e.g. ``{"name": "search_docs", "arguments": {...}}``."""
    if not content or not tool_names:
        return None
    m = _JSON_BLOCK.search(content)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or obj.get("tool") or obj.get("function")
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    if name in tool_names and isinstance(args, dict):
        return ToolCall(name=name, arguments=args, raw_arguments=json.dumps(args))
    return None


class OpenAICompatibleModel:
    """Client for any ``/chat/completions`` endpoint that follows the OpenAI format."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout_s: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.name = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    @classmethod
    def from_provider(cls, provider: str, model: str, **kw: Any) -> OpenAICompatibleModel:
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider {provider!r}; choose from {sorted(PROVIDERS)}")
        base_url, key_env = PROVIDERS[provider]
        api_key = os.environ.get(key_env) if key_env else None
        if key_env and not api_key:
            raise RuntimeError(f"Set the {key_env} environment variable for provider {provider!r}")
        return cls(model=model, base_url=base_url, api_key=api_key, **kw)

    def check_connection(self) -> None:
        """Fail fast with a clear message if the endpoint is unreachable."""
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            r = requests.get(f"{self.base_url}/models", headers=headers, timeout=10)
        except requests.RequestException as exc:
            raise RuntimeError(f"Cannot reach {self.base_url} - is the model server running? "
                               f"({type(exc).__name__})") from exc
        if r.status_code in (401, 403):
            raise RuntimeError(f"{self.base_url} rejected the API key (HTTP {r.status_code})")

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model, "messages": messages, "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        t0 = time.perf_counter()
        for attempt in range(self.max_retries):
            r = requests.post(f"{self.base_url}/chat/completions", json=payload,
                              headers=headers, timeout=self.timeout_s)
            if r.status_code in (429, 500, 502, 503) and attempt < self.max_retries - 1:
                time.sleep(2 ** attempt * 2)  # rate limit / transient error: back off
                continue
            r.raise_for_status()
            break
        data = r.json()
        latency = time.perf_counter() - t0

        msg = data["choices"][0]["message"]
        calls = []
        for tc in msg.get("tool_calls") or []:
            raw = tc["function"].get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except json.JSONDecodeError:
                args = {"__invalid_json__": raw}
            calls.append(ToolCall(name=tc["function"]["name"], arguments=args,
                                  id=tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                                  raw_arguments=raw if isinstance(raw, str) else json.dumps(raw)))
        content = msg.get("content") or ""
        if not calls and tools:
            names = {t["function"]["name"] for t in tools}
            fallback = parse_text_tool_call(content, names)
            if fallback:
                calls, content = [fallback], ""
        usage = data.get("usage") or {}
        return LLMResponse(content=content, tool_calls=calls,
                           prompt_tokens=int(usage.get("prompt_tokens") or 0),
                           completion_tokens=int(usage.get("completion_tokens") or 0),
                           latency_s=latency)


class ScriptedModel:
    """Returns pre-written responses in order (or from a function). For tests."""

    def __init__(self, responses: list[LLMResponse] | Callable[[list, list | None], LLMResponse],
                 name: str = "scripted") -> None:
        self.name = name
        self._responses = responses
        self.calls: list[tuple[list[dict], list[dict] | None]] = []

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        self.calls.append(([dict(m) for m in messages], tools))
        if callable(self._responses):
            return self._responses(messages, tools)
        if not self._responses:
            raise RuntimeError("ScriptedModel ran out of responses")
        return self._responses.pop(0)


class ExtractiveModel:
    """LLM-free baseline.

    * With a ``search_docs`` tool available and no tool result yet: searches for the
      user's question.
    * Otherwise: answers with the first passage found in the context or tool result.
    * Without any context: says it does not know.
    """

    name = "extractive-baseline"

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        question = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        tool_names = {t["function"]["name"] for t in tools or []}
        if "search_docs" in tool_names and not tool_msgs:
            return LLMResponse(tool_calls=[ToolCall("search_docs", {"query": question})])
        source_text = tool_msgs[-1]["content"] if tool_msgs else ""
        if not source_text:
            ctx = re.search(r"<context>(.*?)</context>", messages[0]["content"] + question, re.S)
            source_text = ctx.group(1) if ctx else ""
        passage = _first_passage(source_text)
        if not passage:
            return LLMResponse(content="I don't know - I could not find this in the documents.")
        return LLMResponse(content=passage)


def _first_passage(text: str) -> str:
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("results", [data])
        if isinstance(data, list) and data and isinstance(data[0], dict):
            first = data[0]
            return f"{first.get('text', '')} [{first.get('source', '')}]".strip()
    except (json.JSONDecodeError, TypeError):
        pass
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    return blocks[0] if blocks else ""
