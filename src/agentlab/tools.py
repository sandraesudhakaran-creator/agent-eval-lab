"""Tool registry: turns plain Python functions into tools an LLM can call.

The JSON schema the model sees is built from the function signature and docstring:

    @registry.tool
    def get_model_specs(model: str) -> dict:
        '''Return the technical data of one vehicle model.

        Args:
            model: Model name, e.g. "Arc".
        '''

Type hints become JSON types, parameters without a default become ``required``, and the
``Args:`` section supplies the parameter descriptions. Tool errors are returned to the model
as text instead of crashing the run, so the agent can correct itself.
"""

from __future__ import annotations

import inspect
import json
import re
import types
import typing
from dataclasses import dataclass
from typing import Any, Callable, Literal, get_args, get_origin

_PY_TO_JSON = {str: "string", int: "integer", float: "number", bool: "boolean", dict: "object"}


def _json_type(tp: Any) -> dict:
    origin = get_origin(tp)
    if origin is Literal:
        values = list(get_args(tp))
        return {"type": _PY_TO_JSON.get(type(values[0]), "string"), "enum": values}
    if origin in (list, tuple):
        args = get_args(tp)
        return {"type": "array", "items": _json_type(args[0]) if args else {}}
    if origin is typing.Union or (hasattr(types, "UnionType") and origin is types.UnionType):
        non_none = [a for a in get_args(tp) if a is not type(None)]
        return _json_type(non_none[0]) if non_none else {"type": "string"}
    return {"type": _PY_TO_JSON.get(tp, "string")}


def _arg_docs(doc: str) -> dict[str, str]:
    out: dict[str, str] = {}
    m = re.search(r"Args:\s*\n(.*?)(?:\n\s*\n|\Z)", doc or "", re.S)
    if m:
        for line in m.group(1).splitlines():
            mm = re.match(r"\s*(\w+)\s*(?:\(.*?\))?:\s*(.+)", line)
            if mm:
                out[mm.group(1)] = mm.group(2).strip()
    return out


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable[..., Any]

    @classmethod
    def from_function(cls, func: Callable, name: str | None = None) -> Tool:
        doc = inspect.getdoc(func) or ""
        description = doc.split("\n\n")[0].replace("\n", " ").strip()
        hints = typing.get_type_hints(func)
        arg_docs = _arg_docs(doc)
        props, required = {}, []
        for pname, p in inspect.signature(func).parameters.items():
            schema = _json_type(hints.get(pname, str))
            if pname in arg_docs:
                schema["description"] = arg_docs[pname]
            props[pname] = schema
            if p.default is inspect.Parameter.empty:
                required.append(pname)
        params = {"type": "object", "properties": props, "required": required}
        return cls(name=name or func.__name__, description=description, parameters=params,
                   func=func)

    def schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters}}


class ToolRegistry:
    def __init__(self, max_result_chars: int = 3000) -> None:
        self._tools: dict[str, Tool] = {}
        self.max_result_chars = max_result_chars

    def tool(self, func: Callable | None = None, *, name: str | None = None):
        """Decorator (``@registry.tool``) that registers a function as a tool."""
        def wrap(f: Callable) -> Callable:
            t = Tool.from_function(f, name)
            self._tools[t.name] = t
            return f
        return wrap(func) if func is not None else wrap

    def add(self, func: Callable, name: str | None = None) -> None:
        t = Tool.from_function(func, name)
        self._tools[t.name] = t

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self, only: list[str] | None = None) -> list[dict]:
        return [t.schema() for n, t in self._tools.items() if only is None or n in only]

    def call(self, name: str, arguments: dict) -> str:
        """Execute a tool and return its result as a string for the model."""
        if name not in self._tools:
            return f"ERROR: unknown tool {name!r}. Available tools: {', '.join(self._tools)}"
        if "__invalid_json__" in arguments:
            return "ERROR: tool arguments were not valid JSON. Call the tool again with valid JSON."
        tool = self._tools[name]
        allowed = set(tool.parameters["properties"])
        unknown = set(arguments) - allowed
        missing = [p for p in tool.parameters["required"] if p not in arguments]
        if unknown or missing:
            return (f"ERROR: invalid arguments for {name}. Missing: {missing or 'none'}; "
                    f"unknown: {sorted(unknown) or 'none'}. Expected: {sorted(allowed)}")
        try:
            result = tool.func(**arguments)
        except Exception as exc:  # report to the model, don't crash the agent
            return f"ERROR: {name} failed: {exc}"
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        if len(text) > self.max_result_chars:
            text = text[: self.max_result_chars] + " …[truncated]"
        return text
