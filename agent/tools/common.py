"""Shared helpers for SQLite-backed tools."""

from __future__ import annotations

import sqlite3
from typing import Iterable


UNKNOWN_VALUES = {None, "", "unknown"}


def configure_connection(conn: sqlite3.Connection) -> sqlite3.Connection:
    conn.row_factory = sqlite3.Row
    return conn


def row_to_dict(row: sqlite3.Row | None) -> dict:
    return {} if row is None else dict(row)


def missing_fields(data: dict, required_fields: Iterable[str]) -> list[str]:
    return [field for field in required_fields if data.get(field) in UNKNOWN_VALUES]
