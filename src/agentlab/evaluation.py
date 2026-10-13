"""Evaluation harness: run every test case through every model x mode and score it.

Scoring is rule-based and transparent, so a result can always be traced back to a reason:

* **facts**      - each expected fact must appear in the answer (any accepted spelling;
                   numbers match with or without thousands separators).
* **forbidden**  - strings that signal a hallucination or a wrong action must not appear.
* **refusal**    - for unanswerable questions the answer must say it doesn't know.
* **tools**      - did the agent call the tools a good solution needs (recall), and only
                   useful ones (precision)? For actions: were the arguments right?
* **sources**    - did retrieval/tool use reach the documents that hold the answer?

A case *passes* when its facts are all present, nothing forbidden appears, required tool
arguments match, and no error occurred (for unanswerable cases: refusal + nothing forbidden).
An optional LLM judge can add a second, semantic opinion; the rule-based verdict stays primary.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml

from agentlab.agent import LLMOnly, RAGPipeline, RunResult, ToolAgent
from agentlab.demo import AGENT_PROMPT, SYSTEM_PROMPT, VeloraDomain
from agentlab.llm import ChatModel

MODES = ("llm_only", "rag", "agent")
MODEL_DATA_TOOLS = {"get_model_specs", "compare_models", "estimate_charging_cost"}
REFUSAL_MARKERS = (
    "don't know", "do not know", "not available", "no information", "couldn't find",
    "could not find", "not in the documents", "isn't mentioned", "is not mentioned",
    "not mentioned", "don't have", "do not have", "unable to", "no details", "not specified",
    "no data", "doesn't provide", "does not provide", "not provided", "no record", "unknown",
    "cannot find", "can't find", "not listed", "isn't listed", "there is no", "there's no",
    "not offer", "doesn't offer", "does not exist", "doesn't exist", "not part of",
)


@dataclass
class Case:
    id: str
    category: str
    question: str
    expected_facts: list[list[str]] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    answerable: bool = True
    expected_tools: list[str] = field(default_factory=list)
    expected_tool_args: dict[str, dict] = field(default_factory=dict)
    expected_sources: list[str] = field(default_factory=list)


def load_cases(path: str | Path) -> list[Case]:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cases = [Case(**{k: v for k, v in r.items()}) for r in raw]
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ids")
    return cases


# --- text matching -------------------------------------------------------------------------

_THOUSANDS = re.compile(r"(?<=\d)[,.   ](?=\d{3}(?!\d))")


def normalize(text: str) -> str:
    text = text.lower().replace("‑", "-").replace("–", "-")
    return _THOUSANDS.sub("", text)


_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def contains(answer: str, phrase: str) -> bool:
    """Case-insensitive match. A purely numeric phrase matches the same number anywhere in
    the answer (26.0 == 26.00 == 26); other phrases with digits must not be part of a
    longer number."""
    a, p = normalize(answer), normalize(phrase)
    if re.fullmatch(r"\d+(?:[.,]\d+)?", p.strip()):
        target = float(p.strip().replace(",", "."))
        return any(abs(float(n.replace(",", ".")) - target) < 1e-9 for n in _NUMBER.findall(a))
    if re.search(r"\d", p):
        return re.search(r"(?<![\d])" + re.escape(p) + r"(?![\d])", a) is not None
    return p in a


def is_refusal(answer: str) -> bool:
    a = normalize(answer)
    return any(m in a for m in REFUSAL_MARKERS)


# --- scoring -------------------------------------------------------------------------------

@dataclass
class Score:
    case_id: str
    category: str
    model: str
    mode: str
    passed: bool
    fact_recall: float | None
    facts_missing: list[str]
    forbidden_found: list[str]
    refusal: bool
    tool_recall: float | None
    tool_precision: float | None
    tool_args_ok: bool | None
    source_recall: float | None
    latency_s: float
    tokens: int
    llm_calls: int
    stopped: str
    error: str
    judge: bool | None = None


def score(case: Case, run: RunResult, model_name: str) -> Score:
    answer = run.answer or ""
    facts_missing = [alts[0] for alts in case.expected_facts
                     if not any(contains(answer, a) for a in alts)]
    fact_recall = (1 - len(facts_missing) / len(case.expected_facts)
                   if case.expected_facts else None)
    forbidden = [f for f in case.forbidden if contains(answer, f)]
    refusal = is_refusal(answer)

    called = [s.name for s in run.tool_calls if s.name != "retrieve"]
    tool_recall = tool_precision = None
    if case.expected_tools and run.mode == "agent":
        groups = [set(g.split("|")) for g in case.expected_tools]
        tool_recall = sum(bool(g & set(called)) for g in groups) / len(groups)
        useful = set().union(*groups)
        tool_precision = (sum(c in useful for c in set(called)) / len(set(called))
                          if called else 0.0)

    args_ok = None
    if case.expected_tool_args:
        args_ok = all(
            any(s.name == tool and all(str((s.arguments or {}).get(k, "")).strip().lower()
                                       == str(v).strip().lower() for k, v in want.items())
                for s in run.tool_calls)
            for tool, want in case.expected_tool_args.items())

    sources = set(run.sources)
    if MODEL_DATA_TOOLS & set(called):
        sources.add("models")
    source_recall = None
    if case.expected_sources and run.mode != "llm_only":
        hit = sum(s in sources for s in case.expected_sources)
        source_recall = hit / len(case.expected_sources)

    if run.stopped == "error":
        passed = False
    elif case.answerable:
        passed = not facts_missing and not forbidden and args_ok is not False
    else:
        passed = refusal and not forbidden

    return Score(case.id, case.category, model_name, run.mode, passed, fact_recall,
                 facts_missing, forbidden, refusal, tool_recall, tool_precision, args_ok,
                 source_recall, round(run.latency_s, 3), run.total_tokens, run.n_llm_calls,
                 run.stopped, run.error)


JUDGE_PROMPT = """You grade answers of a customer assistant.
Question: {question}
Facts the answer must contain: {facts}
Is the question answerable from the company's data: {answerable}
Answer to grade: {answer}

Reply with exactly one word: PASS if the answer is correct and contains the facts (or, for an
unanswerable question, clearly says the information is not available without inventing it);
otherwise FAIL."""


def judge(judge_model: ChatModel, case: Case, answer: str) -> bool:
    facts = "; ".join(" / ".join(a) for a in case.expected_facts) or "none"
    msg = JUDGE_PROMPT.format(question=case.question, facts=facts,
                              answerable="yes" if case.answerable else "no", answer=answer)
    resp = judge_model.chat([{"role": "user", "content": msg}])
    return "PASS" in resp.content.upper()


# --- running -------------------------------------------------------------------------------

def build_pipeline(mode: str, model: ChatModel, domain: VeloraDomain, rag_k: int = 4,
                   max_steps: int = 6):
    if mode == "llm_only":
        return LLMOnly(model, SYSTEM_PROMPT)
    if mode == "rag":
        return RAGPipeline(model, domain.index, SYSTEM_PROMPT, k=rag_k)
    if mode == "agent":
        return ToolAgent(model, domain.registry, AGENT_PROMPT, max_steps=max_steps)
    raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")


def run_eval(
    models: list[ChatModel],
    cases: list[Case],
    modes: tuple[str, ...] | list[str] = MODES,
    domain: VeloraDomain | None = None,
    judge_model: ChatModel | None = None,
    rag_k: int = 4,
    max_steps: int = 6,
    log: Callable[[str], None] = print,
) -> tuple[list[Score], list[dict]]:
    """Run all cases; returns scores and full run records (with traces)."""
    domain = domain or VeloraDomain()
    scores, records = [], []
    for model in models:
        for mode in modes:
            pipe = build_pipeline(mode, model, domain, rag_k, max_steps)
            n_pass = 0
            for case in cases:
                domain.bookings.clear()
                try:
                    run = pipe.run(case.question)
                except Exception as exc:  # pipelines catch most errors; this is a safety net
                    run = RunResult(case.question, "", mode, stopped="error",
                                    error=f"{type(exc).__name__}: {exc}")
                s = score(case, run, model.name)
                if judge_model is not None and run.stopped != "error":
                    s.judge = judge(judge_model, case, run.answer)
                n_pass += s.passed
                scores.append(s)
                records.append({"model": model.name, "case_id": case.id, **run.to_dict(),
                                "score": s.__dict__})
            log(f"{model.name:>28} | {mode:<8} | {n_pass}/{len(cases)} passed")
    return scores, records


def _mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 3) if xs else None


def summarize(scores: list[Score]) -> list[dict]:
    """One row per model x mode."""
    rows = []
    keys = list(dict.fromkeys((s.model, s.mode) for s in scores))
    for model, mode in keys:
        g = [s for s in scores if s.model == model and s.mode == mode]
        ans = [s for s in g if s.fact_recall is not None]
        una = [s for s in g if s.category == "unanswerable"]
        lat = sorted(s.latency_s for s in g)
        rows.append({
            "model": model, "mode": mode, "cases": len(g),
            "pass_rate": round(sum(s.passed for s in g) / len(g), 3),
            "fact_recall": _mean([s.fact_recall for s in ans]),
            "refusal_accuracy": _mean([float(s.passed) for s in una]),
            "tool_recall": _mean([s.tool_recall for s in g]),
            "tool_precision": _mean([s.tool_precision for s in g]),
            "source_recall": _mean([s.source_recall for s in g]),
            "judge_pass_rate": _mean([float(s.judge) for s in g if s.judge is not None]),
            "mean_latency_s": _mean([s.latency_s for s in g]),
            "p95_latency_s": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 3),
            "mean_tokens": _mean([float(s.tokens) for s in g]),
            "mean_llm_calls": _mean([float(s.llm_calls) for s in g]),
            "errors": sum(s.stopped == "error" for s in g),
        })
    return rows


def by_category(scores: list[Score]) -> dict[str, dict[str, float]]:
    """``{run_label: {category: pass_rate}}`` where run_label = 'model / mode'."""
    out: dict[str, dict[str, float]] = {}
    for s in scores:
        out.setdefault(f"{s.model} / {s.mode}", {}).setdefault(s.category, [])
        out[f"{s.model} / {s.mode}"][s.category].append(s.passed)
    return {k: {c: round(sum(v) / len(v), 3) for c, v in d.items()} for k, d in out.items()}


def save_results(out_dir: str | Path, scores: list[Score], records: list[dict]) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "runs.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump({"summary": summarize(scores), "by_category": by_category(scores)}, fh,
                  indent=2)
    return out
