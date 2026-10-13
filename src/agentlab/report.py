"""Markdown report and chart for an evaluation run."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from agentlab.evaluation import Score, by_category, summarize  # noqa: E402

# Categorical colours in a fixed order (series keep their colour across charts).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
CATEGORY_ORDER = ["doc_lookup", "spec_lookup", "comparison", "calculation", "multi_hop",
                  "action", "unanswerable"]


def _fmt(v) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _pct(v) -> str:
    return "–" if v is None else f"{v * 100:.0f} %"


def plot_by_category(scores: list[Score], path: str | Path) -> Path:
    data = by_category(scores)
    labels = list(data)
    cats = [c for c in CATEGORY_ORDER if any(c in d for d in data.values())]
    cats += sorted({c for d in data.values() for c in d} - set(cats))
    n = len(labels)
    width = 0.8 / max(n, 1)
    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(cats) + 2), 3.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, label in enumerate(labels):
        xs = [j + (i - (n - 1) / 2) * width for j in range(len(cats))]
        ys = [data[label].get(c, 0) * 100 for c in cats]
        ax.bar(xs, ys, width=width * 0.92, color=PALETTE[i % len(PALETTE)], label=label,
               edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(range(len(cats)), [c.replace("_", " ") for c in cats], color=TEXT_2)
    ax.set_ylabel("cases passed [%]", color=TEXT_2)
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2)
    ax.legend(frameon=False, ncol=min(n, 3), loc="lower left", bbox_to_anchor=(0, 1.0),
              fontsize=8)
    ax.set_title("Pass rate by question category", loc="left", color=TEXT, fontweight="bold",
                 pad=28)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def write_report(scores: list[Score], out_dir: str | Path, title: str = "Evaluation report",
                 notes: str = "") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    chart = plot_by_category(scores, out / "pass_rate_by_category.png")
    rows = summarize(scores)
    lines = [f"# {title}", ""]
    if notes:
        lines += [notes, ""]
    lines += [
        "## Overall", "",
        "| model | mode | pass rate | fact recall | refusal acc. | tool recall | tool precision "
        "| source recall | mean latency [s] | mean tokens | LLM calls | errors |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['mode']} | {_pct(r['pass_rate'])} | {_fmt(r['fact_recall'])} "
            f"| {_pct(r['refusal_accuracy'])} | {_fmt(r['tool_recall'])} "
            f"| {_fmt(r['tool_precision'])} | {_fmt(r['source_recall'])} "
            f"| {_fmt(r['mean_latency_s'])} | {_fmt(r['mean_tokens'])} "
            f"| {_fmt(r['mean_llm_calls'])} | {r['errors']} |")
    if any(r["judge_pass_rate"] is not None for r in rows):
        lines += ["", "LLM-judge pass rate (second opinion): " + ", ".join(
            f"{r['model']} / {r['mode']}: {_pct(r['judge_pass_rate'])}" for r in rows)]

    cat = by_category(scores)
    cats = [c for c in CATEGORY_ORDER if any(c in d for d in cat.values())]
    lines += ["", "## Pass rate by category", "", f"![pass rate by category]({chart.name})", "",
              "| run | " + " | ".join(c.replace('_', ' ') for c in cats) + " |",
              "|---|" + "---:|" * len(cats)]
    for label, d in cat.items():
        lines.append(f"| {label} | " + " | ".join(_pct(d.get(c)) for c in cats) + " |")

    failed = [s for s in scores if not s.passed]
    lines += ["", f"## Failed cases ({len(failed)})", "",
              "| model / mode | case | why |", "|---|---|---|"]
    for s in failed:
        why = []
        if s.error:
            why.append(f"error: {s.error[:80]}")
        if s.facts_missing:
            why.append("missing: " + ", ".join(s.facts_missing))
        if s.forbidden_found:
            why.append("forbidden: " + ", ".join(s.forbidden_found))
        if s.category == "unanswerable" and not s.refusal:
            why.append("did not say it doesn't know")
        if s.tool_args_ok is False:
            why.append("wrong or missing tool arguments")
        lines.append(f"| {s.model} / {s.mode} | {s.case_id} | {'; '.join(why) or '–'} |")
    lines += ["", "Full answers and step-by-step traces: `runs.jsonl`.", ""]
    path = out / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
