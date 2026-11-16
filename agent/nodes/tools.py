"""Serial tool execution for the LangGraph message-based evidence flow."""

from __future__ import annotations

import json
import sqlite3

from langchain_core.messages import ToolMessage

from ..session_state import SessionState
from ..state import State
from ..tool_calling import dispatch_graph_call
from ..tools.manual import ManualSearchService


def make_tool_node(
    conn: sqlite3.Connection,
    session_context: SessionState,
    manual_service: ManualSearchService | None = None,
):
    """Create a node with the application-owned query scope for one session."""

    def tool_node(state: State) -> dict:
        messages = list(state.get("messages", []))
        calls = messages[-1].tool_calls
        if len(calls) != 1:
            # The graph routes one result at a time. Reject a parallel batch
            # before executing any tool so a control signal cannot be lost.
            return {"messages": [ToolMessage(
                content=json.dumps({
                    "ok": False,
                    "status": "tool_error",
                    "error_code": "MULTIPLE_TOOL_CALLS_UNSUPPORTED",
                }),
                name=call["name"],
                tool_call_id=call["id"],
            ) for call in calls]}
        call = calls[0]
        try:
            result = dispatch_graph_call(
                call, state, session_context, conn, manual_service)
        except Exception:
            result = {
                "ok": False, "status": "tool_error",
                "error_code": "TOOL_EXECUTION_ERROR",
                "error_message": "工具执行失败",
            }
        observation = ToolMessage(
            content=json.dumps(result, ensure_ascii=False),
            name=call["name"],
            tool_call_id=call["id"],
        )
        return {
            "messages": [observation],
            "tool_call_count": state["tool_call_count"] + 1,
        }

    return tool_node
