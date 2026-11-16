"""Local Agent execution loop."""

from __future__ import annotations

import copy
import sqlite3
from collections.abc import Callable, Iterable
from typing import Any, Union

from .evidence_gate import check_evidence
from .recorder import TrajectoryRecorder
from .schemas import RunnerResult, ToolAction
from .session_state import SessionState
from .tool_registry import execute_tool
from .tools.manual import ManualSearchService


ActionInput = Union[dict[str, Any], ToolAction]
AgentPolicy = Callable[[SessionState], ActionInput]


class AgentRunner:
    def __init__(self, conn: sqlite3.Connection, max_steps: int = 6, manual_service: ManualSearchService | None = None):
        self.conn = conn
        self.max_steps = max_steps
        self.manual_service = manual_service

    def run(
        self,
        state: SessionState,
        agent_policy: AgentPolicy | Iterable[ActionInput],
        intent: str = "logistics_status",
        recorder: TrajectoryRecorder | None = None,
    ) -> RunnerResult:
        policy = self._policy_from_input(agent_policy)
        errors: list[str] = []
        final_action = None
        gate_result = None

        for _ in range(self.max_steps):
            state_before = copy.deepcopy(state)
            try:
                raw_action = policy(state)
            except StopIteration:
                errors.append("POLICY_EXHAUSTED")
                return RunnerResult("policy_exhausted", final_action, gate_result, errors)
            action_type = raw_action.get("type", "tool_call") if isinstance(raw_action, dict) else raw_action.type
            final_action = raw_action.to_dict() if isinstance(raw_action, ToolAction) else dict(raw_action)

            if action_type == "tool_call":
                action = self._coerce_tool_action(raw_action)
                if state.has_called(action):
                    message = f"Duplicate tool call: {action.tool} {action.arguments}"
                    errors.append(message)
                    if recorder:
                        recorder.record_step(final_action, state_before=state_before, state_after=state, error=message)
                    return RunnerResult("error", final_action, gate_result, errors)

                result = execute_tool(action, state, self.conn, self.manual_service)
                state.add_tool_result(action, result)
                if recorder:
                    recorder.record_step(
                        action.to_dict(),
                        result.to_dict(),
                        state_before=state_before,
                        state_after=state,
                    )
                if result.status in {"partial", "no_match"}:
                    continue
                if not result.ok:
                    return RunnerResult("tool_error", final_action, gate_result, errors)
                continue

            if action_type == "finish":
                gate_result = check_evidence(state, intent)
                if recorder:
                    recorder.record_step(
                        final_action,
                        gate_result.to_dict(),
                        state_before=state_before,
                        state_after=state,
                    )
                return RunnerResult("finished" if gate_result.ready else "blocked_by_gate", final_action, gate_result, errors)

            if action_type == "handoff":
                if recorder:
                    recorder.record_step(final_action, state_before=state_before, state_after=state)
                return RunnerResult("handoff", final_action, gate_result, errors)

            if action_type == "ask_user":
                state.clarification_history.append(final_action)
                if recorder:
                    recorder.record_step(final_action, state_before=state_before, state_after=state)
                return RunnerResult("needs_user", final_action, gate_result, errors)

            message = f"Unsupported action type: {action_type}"
            errors.append(message)
            if recorder:
                recorder.record_step(final_action, state_before=state_before, state_after=state, error=message)
            return RunnerResult("error", final_action, gate_result, errors)

        errors.append("MAX_STEPS_EXCEEDED")
        return RunnerResult("max_steps_exceeded", final_action, gate_result, errors)

    @staticmethod
    def _coerce_tool_action(action: ActionInput) -> ToolAction:
        if isinstance(action, ToolAction):
            return action
        return ToolAction(tool=action["tool"], arguments=dict(action.get("arguments") or {}))

    @staticmethod
    def _policy_from_input(agent_policy: AgentPolicy | Iterable[ActionInput]) -> AgentPolicy:
        if callable(agent_policy):
            return agent_policy
        iterator = iter(agent_policy)

        def next_action(_: SessionState) -> ActionInput:
            return next(iterator)

        return next_action
