"""Trusted per-invocation dependencies and authorization scope."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Callable

from .handoff import HandoffClient
from .tools.manual import ManualSearchService


@dataclass(frozen=True)
class RuntimeContext:
    """Trusted inputs for one run.

    Create ``manual_service`` and ``handoff_client`` at application startup and
    reuse them. ``conversation_id`` identifies the business conversation sent
    to the handoff gateway; it is separate from a checkpointer's ``thread_id``.
    """

    connection_factory: Callable[[], sqlite3.Connection]
    default_snapshot_id: str
    visible_snapshot_ids: tuple[str, ...] | None = None
    manual_service: ManualSearchService | None = None
    handoff_client: HandoffClient | None = None
    conversation_id: str = ""

    def __post_init__(self) -> None:
        if not callable(self.connection_factory):
            raise TypeError("RuntimeContext.connection_factory must be callable")
