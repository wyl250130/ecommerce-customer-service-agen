"""Shared dataclasses for the local Agent simulation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ToolStatus = Literal["found", "partial", "no_match", "forbidden", "conflict", "tool_error"]
EvidenceStatus = Literal["available", "unknown", "missing", "not_found", "error", "conflict"]
ActionType = Literal["tool_call", "finish", "handoff", "ask_user"]


@dataclass
class ToolAction:
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    type: ActionType = "tool_call"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolResult:
    ok: bool
    tool: str
    status: ToolStatus
    data: dict[str, Any] = field(default_factory=dict)
    missing_fields: list[str] = field(default_factory=list)
    evidence_refs: list[dict[str, Any]] = field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Evidence:
    source_type: str
    source_id: str
    data: dict[str, Any] = field(default_factory=dict)
    status: EvidenceStatus = "available"
    snapshot_id: str | None = None
    tool: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GateResult:
    ready: bool
    intent: str
    missing: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunnerResult:
    status: str
    final_action: dict[str, Any] | None
    gate_result: GateResult | None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.gate_result is not None:
            payload["gate_result"] = self.gate_result.to_dict()
        return payload
