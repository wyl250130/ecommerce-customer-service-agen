"""Deterministic trajectory verification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .evidence_gate import check_evidence
from .session_state import SessionState
from .schemas import ToolAction
from .tool_registry import ALLOWED_TOOL_NAMES, forbidden_scope_arguments


@dataclass
class VerificationResult:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "errors": self.errors, "warnings": self.warnings}


def verify_trajectory(trajectory: dict[str, Any], final_state: SessionState, intent: str) -> VerificationResult:
    errors: list[str] = []
    warnings: list[str] = []
    seen_calls: set[tuple[str, str]] = set()
    finish_seen = False

    for index, step in enumerate(trajectory.get("steps", [])):
        action = step.get("action") or {}
        action_type = action.get("type", "tool_call")
        if action_type == "tool_call":
            tool = action.get("tool")
            if tool not in ALLOWED_TOOL_NAMES:
                errors.append(f"step {index}: unknown tool {tool}")
            arguments = action.get("arguments") or {}
            supplied = forbidden_scope_arguments(ToolAction(tool=tool, arguments=arguments))
            if supplied:
                errors.append(f"step {index}: action attempted to pass scope/configuration arguments {sorted(supplied)}")
            key = (tool, repr(sorted(arguments.items())))
            if key in seen_calls:
                errors.append(f"step {index}: duplicate tool call {tool}")
            seen_calls.add(key)
        elif action_type == "finish":
            finish_seen = True
        elif action_type not in {"handoff", "ask_user"}:
            errors.append(f"step {index}: unsupported action type {action_type}")

    if finish_seen:
        gate_result = check_evidence(final_state, intent)
        if not gate_result.ready:
            errors.append(f"finish before sufficient evidence: {gate_result.to_dict()}")
    else:
        warnings.append("trajectory has no finish action")

    return VerificationResult(passed=not errors, errors=errors, warnings=warnings)
