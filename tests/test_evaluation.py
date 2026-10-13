from pathlib import Path

import pytest
import yaml

from agentlab.agent import RunResult, TraceStep
from agentlab.cli import main
from agentlab.evaluation import Case, contains, is_refusal, load_cases, run_eval, score, summarize
from agentlab.llm import ExtractiveModel, LLMResponse, ScriptedModel, ToolCall

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("answer,phrase,ok", [
    ("It costs 1.190 €", "1,190", True),
    ("It costs 1190 €", "1,190", True),
    ("covered for 160 000 km", "160,000", True),
    ("about 26.00 €", "26.0", True),
    ("in 2026", "26 €", False),
    ("range 6100 km", "610", False),
    ("The ARC is best", "Arc", True),
])
def test_contains(answer, phrase, ok):
    assert contains(answer, phrase) is ok


def test_refusal_detection():
    assert is_refusal("Sorry, I don't know the acceleration.")
    assert is_refusal("This information is not available in the documents.")
    assert not is_refusal("The Arc accelerates in 5.1 seconds.")


def _run(answer, tools=(), mode="agent", sources=()):
    steps = [TraceStep("llm", "m")] + [TraceStep("tool", n, a) for n, a in tools]
    return RunResult("q", answer, mode, steps, sources=list(sources))


def test_score_answerable_with_tools_and_sources():
    case = Case("c", "multi_hop", "q", expected_facts=[["64,090"]],
                expected_tools=["get_model_specs|compare_models", "search_docs", "calculate"],
                expected_sources=["models", "charging"])
    s = score(case, _run("Total 64.090 €", [("get_model_specs", {}), ("search_docs", {}),
                                            ("calculate", {})], sources=["charging"]), "m")
    assert s.passed and s.tool_recall == 1.0 and s.tool_precision == 1.0
    assert s.source_recall == 1.0
    s2 = score(case, _run("Total 64,000 €", [("calculate", {}), ("book_test_drive", {})]), "m")
    assert not s2.passed and s2.facts_missing == ["64,090"]
    assert s2.tool_recall == pytest.approx(1 / 3) and s2.tool_precision == 0.5


def test_score_action_arguments_and_forbidden():
    case = Case("b", "action", "q", expected_facts=[["TD-"]],
                expected_tool_args={"book_test_drive": {"model": "Ridge", "city": "Hamburg"}})
    good = _run("Booked: TD-ABC123", [("book_test_drive", {"model": "ridge", "city": "Hamburg"})])
    wrong = _run("Booked: TD-ABC123", [("book_test_drive", {"model": "Ridge", "city": "Berlin"})])
    assert score(case, good, "m").passed
    assert not score(case, wrong, "m").passed
    bad = Case("x", "action", "q", expected_facts=[["cannot"]], forbidden=["TD-"])
    assert not score(bad, _run("I cannot, but here is TD-1"), "m").passed


def test_score_unanswerable_and_errors():
    case = Case("u", "unanswerable", "q", answerable=False, forbidden=[" seconds"])
    assert score(case, _run("I don't know that."), "m").passed
    assert not score(case, _run("It takes 5.2 seconds."), "m").passed
    errored = RunResult("q", "", "agent", stopped="error", error="boom")
    assert not score(case, errored, "m").passed


def test_eval_cases_file_is_consistent():
    cases = load_cases(REPO / "data" / "eval_cases.yaml")
    assert len(cases) >= 25
    cats = {c.category for c in cases}
    assert {"doc_lookup", "calculation", "action", "unanswerable"} <= cats
    raw = yaml.safe_load((REPO / "data" / "eval_cases.yaml").read_text())
    allowed = set(Case.__dataclass_fields__)
    assert all(set(r) <= allowed for r in raw)
    for c in cases:
        assert c.answerable == bool(c.expected_facts) or c.category == "unanswerable"


def test_scripted_perfect_agent_passes_a_calculation_case():
    cases = [c for c in load_cases(REPO / "data" / "eval_cases.yaml") if c.id == "arc_lease_total"]

    def agent(messages, tools):
        n_tool = sum(m["role"] == "tool" for m in messages)
        if tools is None:
            return LLMResponse(content="-")
        if n_tool == 0:
            return LLMResponse(tool_calls=[ToolCall("search_docs", {"query": "Arc leasing"})])
        if n_tool == 1:
            return LLMResponse(tool_calls=[ToolCall("calculate", {"expression": "529*36"})])
        return LLMResponse(content="In total 19,044 € over 36 months [leasing].")

    scores, records = run_eval([ScriptedModel(agent, name="perfect")], cases, ["agent"],
                               log=lambda *_: None)
    assert scores[0].passed and scores[0].tool_recall == 1.0 and scores[0].source_recall == 1.0
    assert records[0]["steps"][1]["name"] == "search_docs"


def test_extractive_baseline_full_run_and_cli(tmp_path):
    cases = load_cases(REPO / "data" / "eval_cases.yaml")
    scores, _ = run_eval([ExtractiveModel()], cases, log=lambda *_: None)
    rows = {r["mode"]: r for r in summarize(scores)}
    # retrieval alone answers the document questions; numbers needing tools/maths fail
    assert rows["rag"]["pass_rate"] > rows["llm_only"]["pass_rate"]
    doc = [s for s in scores if s.mode == "rag" and s.category == "doc_lookup"]
    assert sum(s.passed for s in doc) >= 8

    out = tmp_path / "ev"
    assert main(["eval", "--provider", "extractive", "--out", str(out), "--limit", "5",
                 "--cases", str(REPO / "data" / "eval_cases.yaml")]) == 0
    for f in ("report.md", "runs.jsonl", "summary.json", "pass_rate_by_category.png"):
        assert (out / f).exists(), f
