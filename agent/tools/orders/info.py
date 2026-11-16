"""Order fact tools."""

from __future__ import annotations

import sqlite3

from ...schemas import ToolResult
from ..common import missing_fields, row_to_dict


ORDER_FIELDS = (
    "order_id",
    "order_record_status",
    "order_item_status",
    "payment_status",
    "shipment_status",
    "address_status",
    "address_change_status",
    "order_change_status",
)


def get_order_info(conn: sqlite3.Connection, snapshot_id: str, order_id: str) -> ToolResult:
    if not order_id:
        return ToolResult(
            ok=False,
            tool="get_order_info",
            status="tool_error",
            error_code="MISSING_ORDER_ID",
            error_message="order_id is required.",
        )

    cursor = conn.execute(
        """
        SELECT order_id, order_record_status, order_item_status, payment_status,
               shipment_status, address_status, address_change_status, order_change_status
        FROM orders
        WHERE snapshot_id = ?
          AND order_id = ?
        """,
        (snapshot_id, order_id),
    )
    data = row_to_dict(cursor.fetchone())
    if not data:
        return ToolResult(
            ok=False,
            tool="get_order_info",
            status="no_match",
            error_code="ORDER_NOT_FOUND",
            error_message="No order matched this snapshot and order_id.",
        )

    # In the bundled fact store snapshot_id is the primary key of orders,
    # products and skus: each snapshot describes at most one of each. Keep
    # missing product facts explicit instead of borrowing another snapshot.
    product = row_to_dict(conn.execute(
        "SELECT product_id, product_name FROM products WHERE snapshot_id = ?",
        (snapshot_id,),
    ).fetchone())
    sku = row_to_dict(conn.execute(
        "SELECT sku_id, product_id FROM skus WHERE snapshot_id = ?",
        (snapshot_id,),
    ).fetchone())
    product_id = product.get("product_id") or sku.get("product_id")
    sku_id = (sku.get("sku_id") if not product or
              sku.get("product_id") == product.get("product_id") else None)
    data.update(product_id=product_id, product_name=product.get("product_name"),
                sku_id=sku_id,
                product_info_available=bool(product_id and product.get("product_name")))

    missing = missing_fields(data, ("order_id", "order_record_status", "payment_status", "shipment_status"))
    refs = [{"source": "orders", "snapshot_id": snapshot_id, "order_id": order_id}]
    if product:
        refs.append({"source": "products", "snapshot_id": snapshot_id,
                     "product_id": product.get("product_id")})
    if sku_id:
        refs.append({"source": "skus", "snapshot_id": snapshot_id,
                     "sku_id": sku_id})
    return ToolResult(
        ok=True,
        tool="get_order_info",
        status="partial" if missing else "found",
        data=data,
        missing_fields=missing,
        evidence_refs=refs,
    )
