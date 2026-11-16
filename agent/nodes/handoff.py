"""Transfer a validated handoff request through an injected support gateway."""

from __future__ import annotations

import json

from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime

from ..context import RuntimeContext
from ..handoff import HandoffClient, HandoffReason, HandoffResult
from ..session_state import SessionState
from ..state import State


def make_handoff_node(client: HandoffClient, session_context: SessionState):
    """Create the node; the client may be a real gateway or a local simulator."""

    def handoff_node(state: State) -> dict:
        return _transfer(state, client, session_context.conversation_id)

    return handoff_node


def context_handoff_node(state: State, runtime: Runtime[RuntimeContext]) -> dict:
    """Read the handoff gateway and conversation ID from trusted runtime context."""
    context = runtime.context
    if context.handoff_client is None or not context.conversation_id:
        raise ValueError("Handoff client and conversation_id are required")
    return _transfer(state, context.handoff_client, context.conversation_id)


def _transfer(state: State, client: HandoffClient, conversation_id: str) -> dict:
    if state.get("service_status") == "transferred":
        return {}

    request = None
    request_id = None
    for message in reversed(state.get("messages", [])):
        if isinstance(message, ToolMessage) and message.name == "handoff_to_human":
            try:
                request = json.loads(message.content)
            except (TypeError, ValueError) as exc:
                raise ValueError("Invalid latest handoff request") from exc
            request_id = message.tool_call_id
            break
    if (not isinstance(request, dict) or request.get("ok") is not True
            or request.get("route") != "handoff" or not request_id):
        raise ValueError("No validated handoff_to_human request in messages")

    try:
        result = client.transfer(
            conversation_id=conversation_id,
            reason_code=HandoffReason(request["reason_code"]),
            reason_detail=request["reason_detail"],
            request_id=request_id,
        )
    except Exception:
        result = {"success": False, "handoff_status": "failed", "ticket_id": None}

    confirmed = (
        result.get("success") is True
        and result.get("handoff_status") == "transferred"
        and isinstance(result.get("ticket_id"), str)
        and bool(result["ticket_id"].strip())
    )
    handoff_result: HandoffResult = {
        "success": confirmed,
        "handoff_status": "transferred" if confirmed else "failed",
        "ticket_id": result["ticket_id"] if confirmed else None,
    }
    return {
        "service_status": "transferred" if confirmed else "handoff_failed",
        "handoff_result": handoff_result,
    }
