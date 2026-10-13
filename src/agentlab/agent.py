"""Three ways to answer a question - the configurations the evaluation compares.

* :class:`LLMOnly`   - the model answers from its own knowledge (no documents, no tools).
* :class:`RAGPipeline` - retrieve the top-k passages once, put them in the prompt, answer.
* :class:`ToolAgent` - the model decides itself which tools to call (search, specs lookup,
  calculator, booking ...), possibly several times, before answering.

All three return a :class:`RunResult` with the answer and a full trace (every model call and
tool call with latency and tokens), so runs can be inspected and compared.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from agentlab.llm import ChatModel, LLMResponse
from agentlab.rag import BM25Index, format_context
from agentlab.tools import ToolRegistry


@dataclass
class TraceStep:
    kind: str                    # "llm" | "tool"
    name: str = ""               # model name or tool name
    arguments: dict | None = None
    output: str = ""
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class RunResult:
    question: str
    answer: str
    mode: str
    steps: list[TraceStep] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    stopped: str = "answered"    # answered | max_steps | error
    error: str = ""
    nudged: bool = False         # agent got the "use a tool" reminder

    @property
    def tool_calls(self) -> list[TraceStep]:
        return [s for s in self.steps if s.kind == "tool"]

    @property
    def latency_s(self) -> float:
        return sum(s.latency_s for s in self.steps)

    @property
    def total_tokens(self) -> int:
        return sum(s.prompt_tokens + s.completion_tokens for s in self.steps)

    @property
    def n_llm_calls(self) -> int:
        return sum(s.kind == "llm" for s in self.steps)

    def to_dict(self) -> dict:
        return {
            "question": self.question, "answer": self.answer, "mode": self.mode,
            "stopped": self.stopped, "error": self.error, "sources": self.sources,
            "nudged": self.nudged,
            "steps": [s.__dict__ for s in self.steps],
        }


def _llm_step(model: ChatModel, resp: LLMResponse) -> TraceStep:
    return TraceStep("llm", model.name, output=resp.content, latency_s=resp.latency_s,
                     prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens)


def _sources_in(text: str) -> list[str]:
    """Collect ``source`` fields from a JSON tool result (search results carry them)."""
    found: list[str] = []

    def walk(x) -> None:
        if isinstance(x, dict):
            if isinstance(x.get("source"), str):
                found.append(x["source"])
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    try:
        walk(json.loads(text))
    except (json.JSONDecodeError, TypeError):
        pass
    return found


def _unique(xs: list[str]) -> list[str]:
    return list(dict.fromkeys(xs))


class LLMOnly:
    mode = "llm_only"

    def __init__(self, model: ChatModel, system_prompt: str) -> None:
        self.model, self.system_prompt = model, system_prompt

    def run(self, question: str) -> RunResult:
        msgs = [{"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question}]
        resp = self.model.chat(msgs)
        return RunResult(question, resp.content.strip(), self.mode, [_llm_step(self.model, resp)])


class RAGPipeline:
    mode = "rag"

    def __init__(self, model: ChatModel, index: BM25Index, system_prompt: str, k: int = 4) -> None:
        self.model, self.index, self.system_prompt, self.k = model, index, system_prompt, k

    def run(self, question: str) -> RunResult:
        t0 = time.perf_counter()
        hits = self.index.search(question, self.k)
        retrieval = TraceStep("tool", "retrieve", {"query": question, "k": self.k},
                              output=", ".join(h.chunk.ref for h in hits),
                              latency_s=time.perf_counter() - t0)
        prompt = (f"{self.system_prompt}\n\nAnswer using only the context below. "
                  f"Cite sources in square brackets, e.g. [warranty].\n\n"
                  f"<context>\n{format_context(hits)}\n</context>")
        msgs = [{"role": "system", "content": prompt}, {"role": "user", "content": question}]
        resp = self.model.chat(msgs)
        return RunResult(question, resp.content.strip(), self.mode,
                         [retrieval, _llm_step(self.model, resp)],
                         sources=_unique([h.chunk.source for h in hits]))


class ToolAgent:
    """Tool-calling loop.

    1. Send the conversation plus the tool schemas to the model.
    2. If the model requests tools, run them and append the results as ``tool`` messages.
    3. Repeat until the model answers without a tool call, or ``max_steps`` is reached -
       then the model is asked once more, without tools, to answer with what it has.
    """

    mode = "agent"

    NUDGE = ("You answered without using any tool. Do not guess and do not announce a search: "
             "call the tool that can check this now, then answer from its result.")

    def __init__(self, model: ChatModel, registry: ToolRegistry, system_prompt: str,
                 max_steps: int = 6, nudge: bool = True) -> None:
        self.model, self.registry = model, registry
        self.system_prompt, self.max_steps = system_prompt, max_steps
        self.nudge = nudge

    def run(self, question: str) -> RunResult:
        msgs: list[dict] = [{"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": question}]
        res = RunResult(question, "", self.mode)
        schemas = self.registry.schemas()
        try:
            for _ in range(self.max_steps):
                resp = self.model.chat(msgs, tools=schemas)
                res.steps.append(_llm_step(self.model, resp))
                if not resp.tool_calls:
                    if self.nudge and not res.tool_calls and not res.nudged:
                        # Small models often answer from memory or say "let me search"
                        # without calling anything: give them one explicit reminder.
                        res.nudged = True
                        msgs += [{"role": "assistant", "content": resp.content or ""},
                                 {"role": "user", "content": self.NUDGE}]
                        continue
                    res.answer = resp.content.strip()
                    return res
                msgs.append({"role": "assistant", "content": resp.content or "", "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {
                        "name": tc.name, "arguments": json.dumps(tc.arguments)}}
                    for tc in resp.tool_calls]})
                for tc in resp.tool_calls:
                    t0 = time.perf_counter()
                    out = self.registry.call(tc.name, tc.arguments)
                    res.steps.append(TraceStep("tool", tc.name, tc.arguments, out,
                                               time.perf_counter() - t0))
                    res.sources = _unique(res.sources + _sources_in(out))
                    msgs.append({"role": "tool", "tool_call_id": tc.id, "content": out})
            # Step budget used up: ask for a final answer without offering tools.
            msgs.append({"role": "user", "content":
                         "Answer now with the information you have; do not call more tools."})
            resp = self.model.chat(msgs)
            res.steps.append(_llm_step(self.model, resp))
            res.answer, res.stopped = resp.content.strip(), "max_steps"
        except Exception as exc:
            res.stopped, res.error = "error", f"{type(exc).__name__}: {exc}"
        return res
