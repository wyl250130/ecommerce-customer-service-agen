"""Deterministic evidence sufficiency checks."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, ToolMessage

from .evidence import lookup_field
from .schemas import GateResult


UNKNOWN_VALUES = {None, "", "unknown"}

REQUIRED_FIELDS = {
    "shipping_status": ["order_id", "shipment_status"],
    "logistics_status": ["order_id", "shipment_status", "logistics_status"],
}
FIELD_TO_TOOL = {
    "order_id": "get_order_info",
    "shipment_status": "get_order_info",
    "logistics_status": "get_logistics",
}


def check_evidence(messages, intent: str, state=None) -> GateResult:
    """Check whether current-generation observations cover an intent.

    The two-argument SessionState form remains for the local simulation runner.
    """
    required = REQUIRED_FIELDS.get(intent)
    if required is None:
        return GateResult(
            ready=False,
            intent=intent,
            errors=[f"Unsupported intent: {intent}"],
        )

    if hasattr(messages, "evidence") and not isinstance(messages, (list, tuple)):
        return _check_legacy_evidence(messages, intent)
    if state is None:
        raise TypeError("Graph State is required when checking message evidence")
    messages = list(messages)

    missing: list[str] = []
    unknown: list[str] = []
    for field in required:
        lookup = lookup_field(messages, field, state)
        if not lookup.found:
            missing.append(field)
        elif lookup.value in UNKNOWN_VALUES:
            unknown.append(field)

    errors = _current_relevant_errors(messages, state, missing)
    return GateResult(
        ready=not missing and not unknown and not errors,
        intent=intent,
        missing=missing,
        unknown=unknown,
        errors=errors,
    )


def _current_relevant_errors(messages, state: dict, missing: list[str]) -> list[str]:
    """Report failed current-binding calls only when their required field is missing."""
    needed_tools = {FIELD_TO_TOOL[field] for field in missing}
    if not needed_tools:
        return []
    generation = state.get("evidence_generation", 0)
    current_order = state.get("current_order_id")
    calls = {
        call["id"]: call
        for message in messages if isinstance(message, AIMessage)
        for call in message.tool_calls if call.get("id")
    }
    errors = []
    for message in messages:
        if not isinstance(message, ToolMessage) or message.name not in needed_tools:
            continue
        artifact = message.artifact if isinstance(message.artifact, dict) else {}
        if artifact.get("evidence_generation", 0) != generation:
            continue
        call = calls.get(message.tool_call_id)
        if (current_order and call and call.get("args", {}).get("order_id")
                not in {None, current_order}):
            continue
        try:
            result = json.loads(message.content)
        except (TypeError, ValueError):
            continue
        if (not isinstance(result, dict) or result.get("ok") is not False
                or result.get("tool", message.name) != message.name):
            continue
        error = f"{message.name}: {result.get('error_code') or result.get('status') or 'tool_error'}"
        if error not in errors:
            errors.append(error)
    return errors


def _check_legacy_evidence(state, intent: str) -> GateResult:
    """Keep the standalone simulator's Evidence-list API working."""
    required = REQUIRED_FIELDS[intent]
    missing, unknown = [], []
    for field in required:
        item = next((e for e in reversed(state.evidence) if field in e.data), None)
        if item is None:
            missing.append(field)
        elif item.data[field] in UNKNOWN_VALUES:
            unknown.append(field)
    errors = (["Evidence contains not_found, error, or conflict status."]
              if any(e.status in {"not_found", "error", "conflict"} for e in state.evidence)
              else [])
    return GateResult(not missing and not unknown and not errors, intent,
                      missing, unknown, errors)
