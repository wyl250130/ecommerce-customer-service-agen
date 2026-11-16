"""Build the LLM A -> tools -> LLM B customer-service graph."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from .context import RuntimeContext
from .handoff import HandoffClient
from .nodes.handoff import context_handoff_node, make_handoff_node
from .nodes.llma import make_llma_node
from .nodes.llmb import make_llmb_node
from .nodes.tools import make_tool_node
from .session_state import SessionState
from .state import State, validate_initial_state
from .tool_adapters import ALL_TOOLS
from .tool_calling import tool_definitions
from .tools.manual import ManualSearchService


def should_continue(state: State) -> str:
    """Send LLM A tool calls to the executor; plain text is a clarification."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool"
    return END


def should_continue_with_toolnode(state: State) -> str:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool" if len(last_message.tool_calls) == 1 else "invalid_tool_calls"
    return END


def reject_multiple_tool_calls(state: State) -> dict:
    """Keep ToolNode from running parallel calls that compete to update State."""
    calls = state["messages"][-1].tool_calls
    return {
        "messages": [ToolMessage(
            content=json.dumps({"ok": False, "status": "tool_error",
                                "error_code": "MULTIPLE_TOOL_CALLS_UNSUPPORTED"}),
            name=call["name"], tool_call_id=call["id"],
        ) for call in calls],
        "tool_call_count": state["tool_call_count"] + len(calls),
    }


def route_after_tool(state: State) -> str:
    """Route a validated control result, or let LLM A inspect an observation."""
    last_message = state["messages"][-1]
    if not isinstance(last_message, ToolMessage):
        return "llma"
    try:
        result = json.loads(last_message.content)
    except (TypeError, ValueError):
        return "llma"
    if not isinstance(result, dict) or result.get("ok") is not True:
        return "llma"
    if last_message.name == "finish_collection" and result.get("route") == "respond":
        return "llmb"
    if last_message.name == "handoff_to_human" and result.get("route") == "handoff":
        return "handoff"
    if last_message.name == "request_fallback" and result.get("route") == "fallback":
        return END
    return "llma"


def build_agent(
    llma_model: Any,
    llmb_model: Any,
    conn: sqlite3.Connection | None = None,
    session_context: SessionState | None = None,
    handoff_client: HandoffClient | None = None,
    manual_service: ManualSearchService | None = None,
    checkpointer: Any | None = None,
):
    """Compile the ToolRuntime graph; old connection arguments remain supported.

    New callers invoke the graph with ``context=RuntimeContext(...)``.
    """
    native = conn is None
    if not native and (session_context is None or handoff_client is None):
        raise TypeError("Legacy path requires session_context and handoff_client")
    builder = StateGraph(State, context_schema=RuntimeContext) if native else StateGraph(State)
    bound_tools = ALL_TOOLS if native else tool_definitions()
    builder.add_node("llma", make_llma_node(llma_model.bind_tools(bound_tools)))
    builder.add_node("tool", ToolNode(ALL_TOOLS) if native else
                     make_tool_node(conn, session_context, manual_service))
    if native:
        builder.add_node("validate_state", validate_initial_state)
        builder.add_node("invalid_tool_calls", reject_multiple_tool_calls)
    builder.add_node("llmb", make_llmb_node(llmb_model, session_context))
    builder.add_node("handoff", context_handoff_node if native else
                     make_handoff_node(handoff_client, session_context))

    if native:
        builder.add_edge(START, "validate_state")
        builder.add_edge("validate_state", "llma")
    else:
        builder.add_edge(START, "llma")
    builder.add_conditional_edges(
        "llma", should_continue_with_toolnode if native else should_continue,
        {"tool": "tool", "invalid_tool_calls": "invalid_tool_calls", END: END}
        if native else {"tool": "tool", END: END},
    )
    if native:
        builder.add_edge("invalid_tool_calls", "llma")
    builder.add_conditional_edges(
        "tool", route_after_tool,
        {"llma": "llma", "llmb": "llmb", "handoff": "handoff", END: END},
    )
    builder.add_edge("llmb", END)
    builder.add_edge("handoff", END)
    return builder.compile(checkpointer=checkpointer)
