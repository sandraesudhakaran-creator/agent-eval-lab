"""Demo domain: customer assistant for *Velora Motors*, a fictional electric-car maker.

The knowledge base (``data/kb/*.md``) and the model table (``data/models.csv``) are
invented for this project. The domain is only a test bed: swap in any document folder
and any set of tools to evaluate an agent for a different use case.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import operator
import re
from datetime import date
from pathlib import Path
from typing import Literal

from agentlab.rag import BM25Index, load_markdown_dir
from agentlab.tools import ToolRegistry

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

Attribute = Literal["price_eur", "battery_kwh", "range_wltp_km", "dc_charge_kw",
                    "charge_10_80_min", "seats", "boot_l", "tow_kg"]
STUDIOS = ["Berlin", "Hamburg", "Munich", "Cologne", "Frankfurt"]
CHARGING_EFFICIENCY = 0.90  # share of grid energy that ends up in the battery

SYSTEM_PROMPT = (
    "You are the customer assistant of Velora Motors, a maker of electric cars "
    "(models: Pulse, Arc, Ridge). Answer briefly and precisely. "
    "Only state facts, prices and numbers that come from the provided documents or tools - "
    "never guess. If the information is not available, say clearly that you don't know. "
    "When you use document content, cite the source in square brackets, e.g. [warranty]."
)

AGENT_PROMPT = SYSTEM_PROMPT + (
    " You have tools: use search_docs for policies, prices of services, warranty, charging, "
    "leasing and app questions; get_model_specs or compare_models for technical data and car "
    "prices; calculate or estimate_charging_cost for any arithmetic. Only call book_test_drive "
    "when the user explicitly asks to book and has given model, city and date."
)


def load_models(data_dir: Path = DATA_DIR) -> dict[str, dict]:
    with open(data_dir / "models.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for r in rows:
        out[r["model"].lower()] = {k: _num(v) for k, v in r.items()}
    return out


def _num(v: str):
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
        ast.UAdd: operator.pos, ast.Mod: operator.mod}


def safe_eval(expression: str) -> float:
    """Evaluate +, -, *, /, %, ** and parentheses - no names, calls or attributes."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("only numbers and + - * / % ** ( ) are allowed")
    expr = expression.replace("€", "").replace("^", "**").replace(",", "")
    if len(expr) > 200:
        raise ValueError("expression too long")
    return ev(ast.parse(expr, mode="eval"))


class VeloraDomain:
    """Holds the data and exposes the domain tools through a :class:`ToolRegistry`."""

    def __init__(self, data_dir: str | Path = DATA_DIR) -> None:
        self.data_dir = Path(data_dir)
        self.models = load_models(self.data_dir)
        self.index = BM25Index(load_markdown_dir(self.data_dir / "kb"))
        self.bookings: list[dict] = []
        self.registry = self._build_registry()

    def _model(self, name: str) -> dict:
        key = re.sub(r"^velora\s+", "", name.strip().lower())
        if key not in self.models:
            raise ValueError(f"unknown model {name!r}; Velora models are Pulse, Arc and Ridge")
        return self.models[key]

    def _build_registry(self) -> ToolRegistry:
        reg = ToolRegistry()
        dom = self

        @reg.tool
        def search_docs(query: str, k: int = 3) -> dict:
            """Search Velora's documents (warranty, charging, service, test drives, delivery,
            leasing, software and app) and return the most relevant passages with their source.

            Args:
                query: What to look for, in a few keywords.
                k: Number of passages to return (1-5).
            """
            hits = dom.index.search(query, max(1, min(int(k), 5)))
            return {"results": [{"source": h.chunk.source, "section": h.chunk.section,
                                 "text": h.chunk.text} for h in hits]}

        @reg.tool
        def get_model_specs(model: str) -> dict:
            """Return price and technical data of one Velora model (battery, range, charging
            power, charging time, seats, boot volume, towing capacity, drive).

            Args:
                model: Model name: Pulse, Arc or Ridge.
            """
            return dom._model(model)

        @reg.tool
        def compare_models(attribute: Attribute, models: list[str] | None = None) -> dict:
            """Compare one attribute across Velora models, sorted from highest to lowest
            (for charge_10_80_min the fastest model is the one with the lowest value).

            Args:
                attribute: price_eur, battery_kwh, range_wltp_km, dc_charge_kw,
                    charge_10_80_min (fast-charging time 10-80 % in minutes), seats,
                    boot_l (boot volume in litres) or tow_kg (towing capacity in kg).
                models: Models to include; all models if omitted.
            """
            rows = [dom._model(m) for m in (models or list(dom.models))]
            if attribute not in rows[0]:
                raise ValueError(f"unknown attribute {attribute!r}; options: {list(rows[0])}")
            ranked = sorted(rows, key=lambda r: r[attribute], reverse=True)
            return {"attribute": attribute,
                    "ranking": [{"model": r["model"], attribute: r[attribute]} for r in ranked]}

        @reg.tool
        def calculate(expression: str) -> dict:
            """Evaluate an arithmetic expression exactly, e.g. "529 * 36".

            Args:
                expression: Numbers with + - * / ** and parentheses.
            """
            value = safe_eval(expression)
            return {"expression": expression, "result": round(value, 4)}

        @reg.tool
        def estimate_charging_cost(model: str, start_percent: float, end_percent: float,
                                   price_per_kwh: float = 0.39) -> dict:
            """Estimate energy and cost to charge a Velora model between two charge levels,
            including charging losses.

            Args:
                model: Model name: Pulse, Arc or Ridge.
                start_percent: Battery level before charging, 0-100.
                end_percent: Battery level after charging, 0-100.
                price_per_kwh: Electricity price in euros per kWh. Leave it out to use the
                    Velora Charge card price of 0.39; only set it if the user gives a
                    different price.
            """
            m = dom._model(model)
            if not 0 <= start_percent < end_percent <= 100:
                raise ValueError("need 0 <= start_percent < end_percent <= 100")
            to_battery = m["battery_kwh"] * (end_percent - start_percent) / 100
            from_grid = to_battery / CHARGING_EFFICIENCY
            return {"model": m["model"], "energy_into_battery_kwh": round(to_battery, 2),
                    "energy_from_grid_kwh": round(from_grid, 2),
                    "price_per_kwh_eur": price_per_kwh,
                    "cost_eur": round(from_grid * price_per_kwh, 2),
                    "assumed_charging_efficiency": CHARGING_EFFICIENCY}

        @reg.tool
        def book_test_drive(model: str, city: str, date_iso: str) -> dict:
            """Book a free 45-minute test drive at a Velora studio.

            Args:
                model: Model name: Pulse, Arc or Ridge.
                city: Studio city: Berlin, Hamburg, Munich, Cologne or Frankfurt.
                date_iso: Date in the format YYYY-MM-DD.
            """
            m = dom._model(model)
            studio = next((s for s in STUDIOS if s.lower() == city.strip().lower()), None)
            if studio is None:
                raise ValueError(f"no Velora studio in {city!r}; studios: {', '.join(STUDIOS)}")
            day = date.fromisoformat(date_iso)
            booking_id = "TD-" + hashlib.sha1(
                f"{m['model']}{studio}{day}".encode()).hexdigest()[:6].upper()
            booking = {"booking_id": booking_id, "model": m["model"], "city": studio,
                       "date": day.isoformat(), "duration_min": 45, "status": "confirmed"}
            dom.bookings.append(booking)
            return booking

        return reg
