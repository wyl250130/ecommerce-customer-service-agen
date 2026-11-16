"""Dataset-backed scoped business query."""
import json
from ...schemas import ToolResult
from ..business_common import _row, _error


def get_recent_orders(conn, snapshot_id, visible_snapshot_ids=None, limit=5, product_hint=None):
    tool = "get_recent_orders"
    if type(limit) is not int or not 1 <= limit <= 20 or (product_hint is not None and (not isinstance(product_hint, str) or not product_hint.strip())):
        return _error(tool, "ARGUMENT_ERROR")
    scope = list(dict.fromkeys(visible_snapshot_ids if visible_snapshot_ids is not None else [snapshot_id]))
    orders, refs = [], []
    for sid in scope:
        row = _row(conn.execute("""SELECT o.order_id, p.product_id, p.product_name, s.sku_id,
                    o.payment_status, o.shipment_status
                    FROM orders o LEFT JOIN products p USING(snapshot_id)
                    LEFT JOIN skus s USING(snapshot_id) WHERE o.snapshot_id = ?""", (sid,)))
        if row is None or (product_hint and product_hint.casefold() not in (row.get("product_name") or "").casefold()):
            continue
        orders.append({**row, "order_time": None})
        refs.append({"source": "orders", "snapshot_id": sid, "order_id": row["order_id"]})
    # Stable application-scope ordering, never pretend this is chronological.
    return ToolResult(bool(orders), tool, "partial" if orders else "no_match",
                      {"orders": orders[:limit], "total": len(orders), "has_more": len(orders) > limit,
                       "ordering": "application_scope_not_chronological", "ownership_verified": False},
                      ["order_time", "user_order_ownership"], refs[:limit],
                      error_code=None if orders else "ORDERS_NOT_FOUND")
