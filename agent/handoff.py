"""Reason codes and an injectable human-support gateway."""

from __future__ import annotations

from enum import Enum
from typing import Protocol, TypedDict, runtime_checkable
from uuid import uuid4


class HandoffReason(str, Enum):
    USER_REQUESTED = "user_requested"
    CONFLICTING_FACTS = "conflicting_facts"
    HIGH_RISK_DISPUTE = "high_risk_dispute"
    REFUND_DISPUTE = "refund_dispute"
    PERMISSION_REQUIRED = "permission_required"
    TOOL_UNAVAILABLE = "tool_unavailable"
    OTHER = "other"


class HandoffResult(TypedDict):
    success: bool
    handoff_status: str
    ticket_id: str | None


@runtime_checkable
class HandoffClient(Protocol):
    def transfer(
        self, *, conversation_id: str, reason_code: HandoffReason,
        reason_detail: str, request_id: str,
    ) -> HandoffResult:
        """Send a transfer request; request_id is an idempotency key."""


class SimulatedHandoffClient:
    """Local stand-in for a human-support system; no person is contacted."""

    def __init__(self) -> None:
        self._results: dict[tuple[str, str], HandoffResult] = {}

    def transfer(
        self, *, conversation_id: str, reason_code: HandoffReason,
        reason_detail: str, request_id: str,
    ) -> HandoffResult:
        key = (conversation_id, request_id)
        if key not in self._results:
            self._results[key] = {
                "success": True,
                "handoff_status": "transferred",
                "ticket_id": f"SIM-{uuid4().hex[:12].upper()}",
            }
        return self._results[key].copy()
