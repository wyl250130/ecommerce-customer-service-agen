"""Dataset-backed business queries; scope is supplied by the application."""
import json
import sqlite3

from ..schemas import ToolResult


def _row(cursor):
    value = cursor.fetchone()
    return dict(zip([column[0] for column in cursor.description], value)) if value is not None else None


def _error(tool, code, status="tool_error"):
    return ToolResult(False, tool, status, error_code=code)
