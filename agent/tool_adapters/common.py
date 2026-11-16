"""Bridge injected ToolRuntime to existing guarded business functions."""

from __future__ import annotations

import json

from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from ..context import RuntimeContext
from ..evidence import available_tool_observations
from ..schemas import ToolAction
from ..tool_registry import execute_tool


def _single_call(runtime: ToolRuntime[RuntimeContext]) -> bool:
    messages = runtime.state.get("messages", [])
    return (bool(messages) and isinstance(messages[-1], AIMessage)
            and len(messages[-1].tool_calls) == 1)


def _message(name: str, payload: dict, runtime: ToolRuntime[RuntimeContext], generation: int) -> ToolMessage:
    return ToolMessage(
        content=json.dumps(payload, ensure_ascii=False),
        name=name,
        tool_call_id=runtime.tool_call_id or "missing_call_id",
        artifact={"evidence_generation": generation},
    )


def _command(name: str, payload: dict, runtime: ToolRuntime[RuntimeContext],
             generation: int, **updates) -> Command:
    return Command(update={
        "messages": [_message(name, payload, runtime, generation)],
        "tool_call_count": runtime.state["tool_call_count"] + 1,
        "evidence_generation": generation,
        **updates,
    })


def run_business(name: str, arguments: dict, runtime: ToolRuntime[RuntimeContext]) -> Command:
    """Resolve scope via the existing registry and persist graph-visible changes."""
    state, context = runtime.state, runtime.context
    generation = state["evidence_generation"]
    if not _single_call(runtime):
        return _command(name, {"ok": False, "status": "tool_error",
                               "error_code": "MULTIPLE_TOOL_CALLS_UNSUPPORTED"}, runtime, generation)

    old_binding = (
        state.get("active_snapshot_id") or context.default_snapshot_id,
        state.get("current_order_id"),
        state.get("current_product_id"),
        state.get("current_sku_id"),
    )
    action = ToolAction(name, arguments)
    connection = context.connection_factory()
    try:
        result, state_updates = execute_tool(
            action,
            conn=connection,
            active_snapshot_id=state.get("active_snapshot_id"),
            default_snapshot_id=context.default_snapshot_id,
            visible_snapshot_ids=context.visible_snapshot_ids,
            current_product_id=state.get("current_product_id"),
            current_sku_id=state.get("current_sku_id"),
            manual_service=context.manual_service,
        )
    finally:
        connection.close()
    payload = result.to_dict()
    new_binding = (
        state_updates.get("active_snapshot_id", old_binding[0]),
        state_updates.get("current_order_id", old_binding[1]),
        state_updates.get("current_product_id", old_binding[2]),
        state_updates.get("current_sku_id", old_binding[3]),
    )
    # All old observations, including policy results, expire on a binding change.
    if result.ok and old_binding != new_binding:
        generation += 1

    next_state = {
        **state,
        **state_updates,
        "evidence_generation": generation,
    }
    candidate = _message(name, payload, runtime, generation)
    before = available_tool_observations(state.get("messages", []), next_state)
    after = available_tool_observations([*state.get("messages", []), candidate], next_state)
    if len(after) > len(before):
        payload["evidence_index"] = len(before)

    return _command(name, payload, runtime, generation, **state_updates)


def run_control(name: str, payload: dict, runtime: ToolRuntime[RuntimeContext]) -> Command:
    generation = runtime.state["evidence_generation"]
    if not _single_call(runtime):
        payload = {"ok": False, "status": "tool_error",
                   "error_code": "MULTIPLE_TOOL_CALLS_UNSUPPORTED"}
    return _command(name, payload, runtime, generation)
