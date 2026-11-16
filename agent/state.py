"""LangGraph conversation state and its application-side initializer."""

from typing import Literal

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from typing_extensions import NotRequired

from .context import RuntimeContext
from .handoff import HandoffResult


class State(MessagesState):
    active_snapshot_id: NotRequired[str | None]
    current_order_id: NotRequired[str | None]
    current_product_id: NotRequired[str | None]
    current_sku_id: NotRequired[str | None]
    evidence_generation: NotRequired[int]
    tool_call_count: NotRequired[int]
    service_status: NotRequired[
        Literal["active", "handoff_pending", "transferred", "handoff_failed"]
    ]
    handoff_result: NotRequired[HandoffResult | None]


def make_initial_state(
    messages: list[BaseMessage], *, context: RuntimeContext,
) -> State:
    """Initialize a new conversation from the trusted RuntimeContext."""
    if not isinstance(context, RuntimeContext):
        raise TypeError("context must be a RuntimeContext")
    snapshot_id = context.default_snapshot_id
    if not isinstance(snapshot_id, str) or not snapshot_id.strip():
        raise ValueError("RuntimeContext.default_snapshot_id must be a non-empty string")
    return State(
        messages=messages,
        active_snapshot_id=snapshot_id,
        current_order_id=None,
        current_product_id=None,
        current_sku_id=None,
        evidence_generation=0,
        tool_call_count=0,
        service_status="active",
        handoff_result=None,
    )


def validate_initial_state(state: State) -> dict:
    """Reject an incomplete application input before any model or tool runs."""
    for field in ("tool_call_count", "evidence_generation"):
        value = state.get(field)
        if type(value) is not int or value < 0:
            raise ValueError(
                f"State.{field} must be a non-negative integer; "
                "initialize the graph with make_initial_state()."
            )
    snapshot_id = state.get("active_snapshot_id")
    if not isinstance(snapshot_id, str) or not snapshot_id.strip():
        raise ValueError("State.active_snapshot_id must be initialized from RuntimeContext.default_snapshot_id")
    return {}
