"""Find usable observations in the current LangGraph message history."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

from langchain_core.messages import ToolMessage

BUSINESS_NAMES = frozenset({
    "get_recent_orders", "get_order_info", "get_logistics", "get_product_info",
    "get_after_sales", "search_policies", "search_product_manual",
})


@dataclass(frozen=True)
class FieldLookup:
    field: str
    found: bool
    value: Any = None
    evidence: dict[str, Any] | None = None


def _scope(state: dict | Any, key: str, default=None):
    if isinstance(state, dict):
        return state.get(key, default)
    if key == "active_snapshot_id":
        return getattr(state, "snapshot_id", default)
    return getattr(state, key, default)


def available_tool_observations(messages: Iterable, state: dict | Any) -> list[dict[str, Any]]:
    """Return current-generation business results in message order.

    This ordering is also the numbering used by ``finish_collection``.
    """
    generation = _scope(state, "evidence_generation", 0)
    snapshot = _scope(state, "active_snapshot_id")
    product = _scope(state, "current_product_id")
    sku = _scope(state, "current_sku_id")
    observations = []
    for message in messages:
        if (not isinstance(message, ToolMessage) or message.name not in BUSINESS_NAMES
                or message.name == "get_recent_orders"):
            continue
        artifact = message.artifact if isinstance(message.artifact, dict) else {}
        if artifact.get("evidence_generation", 0) != generation:
            continue
        try:
            result = json.loads(message.content)
        except (TypeError, ValueError):
            continue
        if (not isinstance(result, dict) or result.get("tool") != message.name
                or result.get("ok") is not True
                or result.get("status") not in {"found", "partial"}):
            continue
        data, refs = result.get("data"), result.get("evidence_refs")
        if (not isinstance(data, dict) or not data or not isinstance(refs, list)
                or not refs or any(not isinstance(ref, dict) for ref in refs)):
            continue
        if message.name == "search_policies":
            if any(ref.get("source") != "rule_documents" or not ref.get("policy_id") for ref in refs):
                continue
        elif message.name == "search_product_manual":
            if (not data.get("manual_id") or not product or not sku
                    or data.get("snapshot_id") != snapshot
                    or data.get("product_id") != product or data.get("sku_id") != sku
                    or any(ref.get("source") not in {"frozen_manual_v1", "legacy_category_manual"}
                           or ref.get("manual_id") != data["manual_id"]
                           or (ref.get("source") == "legacy_category_manual" and
                               (ref.get("snapshot_id") != snapshot or
                                ref.get("product_id") != product or ref.get("sku_id") != sku))
                           for ref in refs)):
                continue
        elif any(ref.get("snapshot_id") != snapshot for ref in refs):
            continue
        observations.append(result)
    return observations


def lookup_field(messages: Iterable, field: str, state: dict | Any) -> FieldLookup:
    """Find a field in the newest usable observation."""
    for observation in reversed(available_tool_observations(messages, state)):
        if field in observation["data"]:
            return FieldLookup(field, True, observation["data"][field], observation)
    return FieldLookup(field, False)


def collect_fields(messages: Iterable, fields: Iterable[str], state: dict | Any) -> dict[str, Any]:
    return {field: lookup_field(messages, field, state).value for field in fields}


def collect_field(messages: Iterable, field: str, state: dict | Any):
    return lookup_field(messages, field, state).value
