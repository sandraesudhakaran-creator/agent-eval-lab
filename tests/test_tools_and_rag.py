from typing import Literal

import pytest

from agentlab.demo import VeloraDomain, safe_eval
from agentlab.rag import BM25Index, chunk_markdown, tokenize
from agentlab.tools import Tool, ToolRegistry


def test_schema_from_signature_and_docstring():
    def lookup(model: str, unit: Literal["km", "mi"] = "km", tags: list[str] | None = None) -> dict:
        """Look up a model.

        Args:
            model: Model name.
            unit: Distance unit.
        """
        return {}

    t = Tool.from_function(lookup)
    assert t.description == "Look up a model."
    props = t.parameters["properties"]
    assert props["model"] == {"type": "string", "description": "Model name."}
    assert props["unit"]["enum"] == ["km", "mi"]
    assert props["tags"]["type"] == "array"
    assert t.parameters["required"] == ["model"]
    assert t.schema()["function"]["name"] == "lookup"


def test_registry_reports_errors_instead_of_raising():
    reg = ToolRegistry(max_result_chars=20)

    @reg.tool
    def boom(x: int) -> int:
        """Fail."""
        raise ValueError("bad x")

    @reg.tool
    def long_text() -> str:
        """Long."""
        return "y" * 100

    assert reg.call("nope", {}).startswith("ERROR: unknown tool")
    assert "Missing: ['x']" in reg.call("boom", {})
    assert "unknown: ['z']" in reg.call("boom", {"x": 1, "z": 2})
    assert reg.call("boom", {"x": 1}) == "ERROR: boom failed: bad x"
    assert reg.call("boom", {"__invalid_json__": "{"}).startswith("ERROR: tool arguments")
    assert reg.call("long_text", {}).endswith("[truncated]")


def test_chunking_keeps_heading_path():
    md = "# Top\nintro\n## Sub A\nalpha text\n## Sub B\nbeta text\n"
    chunks = chunk_markdown("doc", md)
    assert [c.section for c in chunks] == ["Top", "Top > Sub A", "Top > Sub B"]
    assert chunks[1].text == "alpha text"


def test_bm25_ranks_relevant_chunk_first():
    md = "# A\nThe wallbox costs money.\n# B\nBattery warranty is eight years.\n# C\nApp features."
    idx = BM25Index(chunk_markdown("d", md))
    assert idx.search("battery warranty")[0].chunk.section == "B"
    assert idx.search("zzz unknown") == []
    assert "the" not in tokenize("The battery")


@pytest.fixture(scope="module")
def dom():
    return VeloraDomain()


@pytest.mark.parametrize("query,source", [
    ("battery warranty capacity", "warranty"),
    ("HomeCharge wallbox installation price", "charging"),
    ("service interval", "service"),
    ("home test drive", "test_drive_and_delivery"),
    ("extra kilometre leasing", "leasing"),
    ("where is data stored", "software_and_app"),
])
def test_retrieval_on_demo_kb(dom, query, source):
    assert dom.index.search(query, 3)[0].chunk.source == source


def test_domain_tools(dom):
    reg = dom.registry
    assert set(reg.names) == {"search_docs", "get_model_specs", "compare_models", "calculate",
                              "estimate_charging_cost", "book_test_drive"}
    assert '"range_wltp_km": 610' in reg.call("get_model_specs", {"model": "Velora Arc"})
    ranking = reg.call("compare_models", {"attribute": "tow_kg"})
    assert ranking.index("Ridge") < ranking.index("Arc") < ranking.index("Pulse")
    assert '"cost_eur": 26.0' in reg.call(
        "estimate_charging_cost", {"model": "Ridge", "start_percent": 20, "end_percent": 80})
    assert '"cost_eur": 17.59' in reg.call(
        "estimate_charging_cost", {"model": "pulse", "start_percent": 10, "end_percent": 80})
    assert '"result": 19044' in reg.call("calculate", {"expression": "529 * 36"})
    assert "unknown model" in reg.call("get_model_specs", {"model": "Nova"})


def test_booking_validates_city_and_date(dom):
    dom.bookings.clear()
    ok = dom.registry.call("book_test_drive",
                           {"model": "Ridge", "city": "hamburg", "date_iso": "2026-11-20"})
    assert '"status": "confirmed"' in ok and "TD-" in ok
    assert dom.bookings[0]["city"] == "Hamburg"
    assert "no Velora studio" in dom.registry.call(
        "book_test_drive", {"model": "Pulse", "city": "Stuttgart", "date_iso": "2026-11-22"})
    assert "ERROR" in dom.registry.call(
        "book_test_drive", {"model": "Pulse", "city": "Berlin", "date_iso": "next friday"})


def test_safe_eval_blocks_code():
    assert safe_eval("(62900 + 1190)") == 64090
    assert safe_eval("2^3") == 8
    for bad in ["__import__('os')", "open('x')", "a + 1"]:
        with pytest.raises(ValueError):
            safe_eval(bad)


def test_arguments_are_coerced_to_schema_types(dom):
    out = dom.registry.call("estimate_charging_cost", {
        "model": "Ridge", "start_percent": "20", "end_percent": "80", "price_per_kwh": "0,39"})
    assert '"cost_eur": 26.0' in out


def test_compare_models_attribute_is_an_enum(dom):
    schema = next(s for s in dom.registry.schemas() if s["function"]["name"] == "compare_models")
    attr = schema["function"]["parameters"]["properties"]["attribute"]
    assert "tow_kg" in attr["enum"] and "model" not in attr["enum"]


def test_argument_error_shows_example_call(dom):
    out = dom.registry.call("search_docs", {"service": "HomeCharge"})
    assert 'Call search_docs again, e.g. with {"query": "..."}' in out


def test_multiline_argument_docs_are_joined():
    def f(a: str, b: int = 1) -> None:
        """Do it.

        Args:
            a: First line
                continues here.
            b: Second.
        """

    props = Tool.from_function(f).parameters["properties"]
    assert props["a"]["description"] == "First line continues here."
    assert props["b"]["description"] == "Second."
