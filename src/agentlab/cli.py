"""Command line.

    agentlab ask "How long is the battery warranty?" --provider ollama --model qwen2.5:7b
    agentlab eval --provider ollama --model qwen2.5:7b --model llama3.1:8b --out outputs/run1
    agentlab eval --provider extractive --out outputs/baseline      # no LLM needed
"""

from __future__ import annotations

import argparse
import json
import sys

from agentlab.demo import VeloraDomain
from agentlab.evaluation import MODES, build_pipeline, load_cases, run_eval, save_results
from agentlab.llm import PROVIDERS, ChatModel, ExtractiveModel, OpenAICompatibleModel
from agentlab.report import write_report

DEFAULT_CASES = "data/eval_cases.yaml"


def make_model(provider: str, model: str | None, base_url: str | None,
               temperature: float) -> ChatModel:
    if provider == "extractive":
        return ExtractiveModel()
    if base_url:
        import os
        return OpenAICompatibleModel(model or "default", base_url=base_url,
                                     api_key=os.environ.get("AGENTLAB_API_KEY"),
                                     temperature=temperature)
    if not model:
        raise SystemExit("--model is required for this provider")
    return OpenAICompatibleModel.from_provider(provider, model, temperature=temperature)


def _add_model_args(p: argparse.ArgumentParser, multi: bool) -> None:
    p.add_argument("--provider", default="ollama", choices=sorted(PROVIDERS) + ["extractive"])
    p.add_argument("--model", action="append" if multi else "store", default=None,
                   help="model name" + (" (repeat to compare models)" if multi else ""))
    p.add_argument("--base-url", default=None,
                   help="any OpenAI-compatible endpoint (key from AGENTLAB_API_KEY)")
    p.add_argument("--temperature", type=float, default=0.0)


def _cmd_ask(a: argparse.Namespace) -> int:
    model = make_model(a.provider, a.model, a.base_url, a.temperature)
    run = build_pipeline(a.mode, model, VeloraDomain()).run(a.question)
    if a.trace:
        for s in run.steps:
            if s.kind == "tool":
                print(f"  -> {s.name}({json.dumps(s.arguments, ensure_ascii=False)})")
                print(f"     {s.output[:300]}")
            else:
                print(f"  [{s.name}] {s.latency_s:.2f}s, "
                      f"{s.prompt_tokens + s.completion_tokens} tokens")
        print()
    print(run.answer or f"(no answer: {run.stopped} {run.error})")
    return 0


def _cmd_eval(a: argparse.Namespace) -> int:
    names = a.model or [None]
    models = [make_model(a.provider, n, a.base_url, a.temperature) for n in names]
    judge = (make_model(a.judge_provider or a.provider, a.judge_model, None, 0.0)
             if a.judge_model else None)
    for m in models + ([judge] if judge else []):
        if hasattr(m, "check_connection"):
            try:
                m.check_connection()
            except RuntimeError as exc:
                raise SystemExit(f"Error: {exc}") from None
    cases = load_cases(a.cases)
    if a.categories:
        cases = [c for c in cases if c.category in a.categories]
    if a.limit:
        cases = cases[: a.limit]
    print(f"Running {len(cases)} cases x {len(models)} model(s) x {len(a.modes)} mode(s)")
    scores, records = run_eval(models, cases, a.modes, judge_model=judge,
                               rag_k=a.rag_k, max_steps=a.max_steps, nudge=not a.no_nudge)
    out = save_results(a.out, scores, records)
    report = write_report(scores, out, title=a.title,
                          notes=f"Provider: `{a.provider}` · cases: {len(cases)} · "
                                f"modes: {', '.join(a.modes)}")
    print(f"\nReport: {report}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agentlab", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="answer one question")
    ask.add_argument("question")
    ask.add_argument("--mode", choices=MODES, default="agent")
    ask.add_argument("--trace", action="store_true", help="print every step")
    _add_model_args(ask, multi=False)
    ask.set_defaults(func=_cmd_ask)

    ev = sub.add_parser("eval", help="run the evaluation set")
    _add_model_args(ev, multi=True)
    ev.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    ev.add_argument("--cases", default=DEFAULT_CASES)
    ev.add_argument("--categories", nargs="*", default=None)
    ev.add_argument("--limit", type=int, default=None)
    ev.add_argument("--rag-k", type=int, default=4)
    ev.add_argument("--max-steps", type=int, default=6)
    ev.add_argument("--no-nudge", action="store_true",
                    help="agent: no reminder when it answers without using a tool (v1 behaviour)")
    ev.add_argument("--judge-model", default=None, help="optional LLM-as-judge model")
    ev.add_argument("--judge-provider", default=None)
    ev.add_argument("--out", default="outputs/eval")
    ev.add_argument("--title", default="Evaluation report")
    ev.set_defaults(func=_cmd_eval)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
