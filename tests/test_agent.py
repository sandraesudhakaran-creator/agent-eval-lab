import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agentlab.agent import LLMOnly, RAGPipeline, ToolAgent
from agentlab.demo import AGENT_PROMPT, SYSTEM_PROMPT, VeloraDomain
from agentlab.llm import (
    ExtractiveModel,
    LLMResponse,
    OpenAICompatibleModel,
    ScriptedModel,
    ToolCall,
    parse_text_tool_call,
)


@pytest.fixture(scope="module")
def dom():
    return VeloraDomain()


def test_agent_multi_step_tool_use(dom):
    model = ScriptedModel([
        LLMResponse(tool_calls=[ToolCall("get_model_specs", {"model": "Ridge"})],
                    prompt_tokens=100, completion_tokens=10, latency_s=0.5),
        LLMResponse(tool_calls=[ToolCall("search_docs", {"query": "wallbox price"})]),
        LLMResponse(tool_calls=[ToolCall("calculate", {"expression": "62900 + 1190"})]),
        LLMResponse(content="Together they cost 64,090 € [charging].", prompt_tokens=300,
                    completion_tokens=20, latency_s=0.7),
    ])
    res = ToolAgent(model, dom.registry, AGENT_PROMPT).run("Ridge plus wallbox?")
    assert res.answer == "Together they cost 64,090 € [charging]."
    assert [s.name for s in res.tool_calls] == ["get_model_specs", "search_docs", "calculate"]
    assert "charging" in res.sources
    assert res.n_llm_calls == 4 and res.total_tokens == 430 and res.latency_s >= 1.2
    # the tool results were fed back to the model in OpenAI message format
    last_msgs = model.calls[-1][0]
    assert last_msgs[-1]["role"] == "tool" and '"result": 64090' in last_msgs[-1]["content"]
    assert last_msgs[-2]["tool_calls"][0]["function"]["name"] == "calculate"


def test_agent_stops_at_max_steps(dom):
    def always_search(messages, tools):
        if tools is None:
            return LLMResponse(content="best effort answer")
        return LLMResponse(tool_calls=[ToolCall("search_docs", {"query": "x"})])

    res = ToolAgent(ScriptedModel(always_search), dom.registry, AGENT_PROMPT, max_steps=3).run("q")
    assert res.stopped == "max_steps" and res.answer == "best effort answer"
    assert len(res.tool_calls) == 3


def test_agent_survives_model_errors(dom):
    def broken(messages, tools):
        raise ConnectionError("model down")

    res = ToolAgent(ScriptedModel(broken), dom.registry, AGENT_PROMPT).run("q")
    assert res.stopped == "error" and "model down" in res.error


def test_rag_puts_context_in_prompt(dom):
    model = ScriptedModel([LLMResponse(content="1,190 € [charging]")])
    res = RAGPipeline(model, dom.index, SYSTEM_PROMPT, k=2).run("wallbox price")
    prompt = model.calls[0][0][0]["content"]
    assert "<context>" in prompt and "1,190" in prompt
    assert res.sources[0] == "charging" and res.tool_calls[0].name == "retrieve"


def test_llm_only_sends_no_tools(dom):
    model = ScriptedModel([LLMResponse(content="I don't know.")])
    LLMOnly(model, SYSTEM_PROMPT).run("q")
    assert model.calls[0][1] is None


def test_extractive_baseline_uses_search(dom):
    res = ToolAgent(ExtractiveModel(), dom.registry, AGENT_PROMPT).run("battery warranty")
    assert res.tool_calls[0].name == "search_docs" and "160,000" in res.answer


def test_parse_text_tool_call():
    names = {"search_docs"}
    tc = parse_text_tool_call('Sure: {"name": "search_docs", "arguments": {"query": "x"}}', names)
    assert tc.name == "search_docs" and tc.arguments == {"query": "x"}
    assert parse_text_tool_call('{"name": "rm_rf", "arguments": {}}', names) is None
    assert parse_text_tool_call("no json here", names) is None


class _FakeOpenAI(BaseHTTPRequestHandler):
    """Minimal /chat/completions endpoint: first a tool call, then an answer."""
    requests_seen: list = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _FakeOpenAI.requests_seen.append((body, self.headers.get("Authorization")))
        if not any(m["role"] == "tool" for m in body["messages"]):
            msg = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "get_model_specs", "arguments": '{"model": "Arc"}'}}]}
        else:
            msg = {"role": "assistant", "content": "The Arc has a range of 610 km."}
        out = json.dumps({"choices": [{"message": msg}],
                          "usage": {"prompt_tokens": 50, "completion_tokens": 5}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args):
        pass


def test_openai_compatible_client_end_to_end(dom):
    server = HTTPServer(("127.0.0.1", 0), _FakeOpenAI)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        model = OpenAICompatibleModel("fake-model", f"http://127.0.0.1:{server.server_port}/v1",
                                      api_key="secret")
        res = ToolAgent(model, dom.registry, AGENT_PROMPT).run("Arc range?")
    finally:
        server.shutdown()
    assert res.answer == "The Arc has a range of 610 km."
    assert res.tool_calls[0].arguments == {"model": "Arc"}
    assert res.total_tokens == 110
    first_body, auth = _FakeOpenAI.requests_seen[0]
    assert auth == "Bearer secret" and first_body["tools"][0]["type"] == "function"
    assert first_body["model"] == "fake-model" and first_body["temperature"] == 0.0
