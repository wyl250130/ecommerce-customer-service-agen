"""Dataset-backed scoped business query."""
import json
from ...schemas import ToolResult
from ..business_common import _row, _error


def get_after_sales(conn, snapshot_id, order_id):
    tool = "get_after_sales"
    if not isinstance(order_id, str) or not order_id.strip():
        return _error(tool, "MISSING_ORDER_ID")
    row = _row(conn.execute("SELECT * FROM after_sales WHERE snapshot_id = ? AND order_id = ?", (snapshot_id, order_id)))
    if row is None:
        return _error(tool, "AFTER_SALES_NOT_FOUND", "no_match")
    row.pop("snapshot_id")
    row["extra"] = json.loads(row.pop("extra_json"))
    missing = [k for k, v in row.items() if v in (None, "", "unknown")]
    return ToolResult(True, tool, "partial" if missing else "found", row, missing,
                      [{"source": "after_sales", "snapshot_id": snapshot_id, "order_id": order_id}])
