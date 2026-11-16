"""Business context used by the local runner and tool adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .schemas import Evidence, ToolAction, ToolResult


@dataclass
class SessionState:
    conversation_id: str
    snapshot_id: str
    current_order_id: str | None = None
    current_product_id: str | None = None
    current_sku_id: str | None = None
    user_messages: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    tool_history: list[dict[str, Any]] = field(default_factory=list)
    clarification_history: list[dict[str, Any]] = field(default_factory=list)
    visible_snapshot_ids: list[str] | None = None

    def has_called(self, action: ToolAction) -> bool:
        return any(
            entry["action"]["tool"] == action.tool
            and entry["action"]["arguments"] == action.arguments
            for entry in self.tool_history
        )

    def add_tool_result(self, action: ToolAction, result: ToolResult) -> None:
        self.tool_history.append({
            "action": action.to_dict(),
            "result": result.to_dict(),
        })
        if not result.ok or not result.evidence_refs:
            return

        if action.tool == "get_product_info":
            self.current_product_id = result.data.get("product_id") or self.current_product_id
            self.current_sku_id = result.data.get("sku_id") or self.current_sku_id

        source_types = {
            "orders": "order",
            "frozen_manual_v1": "product_manual",
            "legacy_category_manual": "product_manual",
            "rule_documents": "policy",
        }
        status = "available" if result.status == "found" else "unknown"
        for ref in result.evidence_refs:
            source = ref.get("source", action.tool)
            self.evidence.append(Evidence(
                source_type=source_types.get(source, source),
                source_id=str(
                    ref.get("order_id") or ref.get("policy_id")
                    or ref.get("manual_id") or self.snapshot_id
                ),
                data=dict(result.data),
                status=status,
                snapshot_id=ref.get("snapshot_id", self.snapshot_id),
                tool=action.tool,
            ))

    def to_visible_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
