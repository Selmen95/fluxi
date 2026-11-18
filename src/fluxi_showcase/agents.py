"""Tool-using agent orchestrator.

The loop is model-agnostic: a Planner looks at the conversation so far and returns
either a ToolCall or a Final answer. The orchestrator owns everything that must be
deterministic and safe: argument validation, an allow-list of tools, a step budget,
an approval gate on side-effecting tools, and a full trace.

`ScriptedPlanner` is a rule-based planner so the demo and tests run offline. A
hosted-LLM planner would implement the same one-method interface.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import pandas as pd

from . import invoices as inv_mod
from . import replenishment as rep_mod


@dataclass
class ToolSpec:
    name: str
    description: str
    params: dict[str, type]
    fn: Callable[..., Any]
    side_effect: bool = False  # True means the call needs approval before it runs


@dataclass
class ToolCall:
    tool: str
    args: dict[str, Any]


@dataclass
class Final:
    answer: str


@dataclass
class TraceStep:
    step: int
    tool: str
    args: dict
    status: str   # ok | error | pending_approval | rejected
    summary: str


class Planner(Protocol):
    def next(self, goal: str, history: list[TraceStep]) -> ToolCall | Final: ...


class Orchestrator:
    def __init__(self, tools: list[ToolSpec], planner: Planner, max_steps: int = 8,
                 approver: Callable[[ToolCall], bool] | None = None):
        self.tools = {t.name: t for t in tools}
        self.planner = planner
        self.max_steps = max_steps
        self.approver = approver  # None means side-effecting calls are never executed
        self.trace: list[TraceStep] = []
        self.results: dict[str, Any] = {}

    def run(self, goal: str) -> str:
        for step in range(1, self.max_steps + 1):
            action = self.planner.next(goal, self.trace)
            if isinstance(action, Final):
                return action.answer
            self.trace.append(self._execute(step, action))
        return "Stopped: step budget exhausted."

    def _execute(self, step: int, call: ToolCall) -> TraceStep:
        spec = self.tools.get(call.tool)
        if spec is None:
            return TraceStep(step, call.tool, call.args, "error", "unknown tool")
        err = self._validate(spec, call.args)
        if err:
            return TraceStep(step, call.tool, call.args, "error", err)
        if spec.side_effect:
            if self.approver is None:
                return TraceStep(step, call.tool, call.args, "pending_approval", "awaiting human approval")
            if not self.approver(call):
                return TraceStep(step, call.tool, call.args, "rejected", "approval refused")
        try:
            out = spec.fn(**call.args)
        except Exception as exc:  # tool failures are data for the planner, not crashes
            return TraceStep(step, call.tool, call.args, "error", f"{type(exc).__name__}: {exc}")
        self.results[call.tool] = out
        size = len(out) if hasattr(out, "__len__") else 1
        return TraceStep(step, call.tool, call.args, "ok", f"{size} item(s)")

    @staticmethod
    def _validate(spec: ToolSpec, args: dict) -> str | None:
        missing = set(spec.params) - set(args)
        extra = set(args) - set(spec.params)
        if missing or extra:
            return f"bad arguments (missing={sorted(missing)}, unexpected={sorted(extra)})"
        for k, t in spec.params.items():
            if not isinstance(args[k], t):
                return f"argument {k} must be {t.__name__}"
        return None


class ScriptedPlanner:
    """Fixed policy: replenishment goal -> propose, then draft; invoice goal -> check."""

    def next(self, goal: str, history: list[TraceStep]) -> ToolCall | Final:
        done = {s.tool: s for s in history if s.status in ("ok", "pending_approval", "rejected")}
        g = goal.lower()
        if "invoice" in g:
            if "check_invoices" not in done:
                return ToolCall("check_invoices", {})
            return Final(f"Invoice control finished: {done['check_invoices'].summary} flagged.")
        if "reorder" in g or "replenish" in g:
            if "propose_replenishment" not in done:
                return ToolCall("propose_replenishment", {"service_level": 0.95})
            if "draft_purchase_orders" not in done:
                return ToolCall("draft_purchase_orders", {"supplier_filter": ""})
            s = done["draft_purchase_orders"]
            return Final(f"Replenishment proposal ready. Draft purchase orders: {s.status}.")
        return Final("No tool matches this goal.")


def build_tools(world, sales: pd.DataFrame) -> list[ToolSpec]:
    state: dict[str, Any] = {}

    def propose_replenishment(service_level: float):
        state["proposal"] = rep_mod.propose_orders(sales, world.stock, world.skus, world.suppliers,
                                                   service_level=service_level)
        return state["proposal"]

    def draft_purchase_orders(supplier_filter: str):
        p = state["proposal"]
        p = p[p["released"]]
        if supplier_filter:
            p = p[p["supplier"] == supplier_filter]
        return p.groupby("supplier")["line_value"].sum().round(2).to_dict()

    def check_invoices():
        return inv_mod.check_invoices(world.invoices, world.purchase_orders)

    return [
        ToolSpec("propose_replenishment", "Compute reorder proposals from demand and stock.",
                 {"service_level": float}, propose_replenishment),
        ToolSpec("draft_purchase_orders", "Create draft purchase orders from the proposal.",
                 {"supplier_filter": str}, draft_purchase_orders, side_effect=True),
        ToolSpec("check_invoices", "Match supplier invoices against purchase orders.",
                 {}, check_invoices),
    ]
