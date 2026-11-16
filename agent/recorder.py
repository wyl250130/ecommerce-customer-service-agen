"""Trajectory recording for local Agent rollouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .session_state import SessionState


@dataclass
class TrajectoryRecorder:
    trajectory_id: str
    started_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    steps: list[dict[str, Any]] = field(default_factory=list)
    final_status: str | None = None

    def record_step(
        self,
        action: dict[str, Any],
        observation: dict[str, Any] | None = None,
        state_before: SessionState | None = None,
        state_after: SessionState | None = None,
        error: str | None = None,
    ) -> None:
        before = state_before.to_visible_dict() if state_before is not None else None
        after = state_after.to_visible_dict() if state_after is not None else None
        self.steps.append(
            {
                "visible_state": before,
                "state_before": before,
                "action": action,
                "observation": observation,
                "state_after": after,
                "error": error,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
