"""Logistics fact tools."""

from __future__ import annotations

import sqlite3

from ...schemas import ToolResult
from ..common import missing_fields, row_to_dict


def get_logistics(conn: sqlite3.Connection, snapshot_id: str, order_id: str) -> ToolResult:
    if not order_id:
        return ToolResult(
            ok=False,
            tool="get_logistics",
            status="tool_error",
            error_code="MISSING_ORDER_ID",
            error_message="order_id is required.",
        )

    cursor = conn.execute(
        """
        SELECT order_id, tracking_no, actual_carrier, logistics_status,
               estimated_delivery, arrival_eta, dispatch_commitment
        FROM logistics
        WHERE snapshot_id = ?
          AND order_id = ?
        """,
        (snapshot_id, order_id),
    )
    data = row_to_dict(cursor.fetchone())
    if not data:
        return ToolResult(
            ok=False,
            tool="get_logistics",
            status="no_match",
            error_code="LOGISTICS_NOT_FOUND",
            error_message="No logistics record matched this snapshot and order_id.",
        )

    missing = missing_fields(data, ("order_id", "logistics_status"))
    return ToolResult(
        ok=True,
        tool="get_logistics",
        status="partial" if missing else "found",
        data=data,
        missing_fields=missing,
        evidence_refs=[{"source": "logistics", "snapshot_id": snapshot_id, "order_id": order_id}],
    )
