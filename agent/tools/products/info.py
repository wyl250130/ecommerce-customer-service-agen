"""Dataset-backed scoped business query."""
import json
from ...schemas import ToolResult
from ..business_common import _row, _error


def get_product_info(conn, snapshot_id, product_id=None, sku_id=None):
    tool = "get_product_info"
    if not any(isinstance(x, str) and x.strip() for x in (product_id, sku_id)):
        return _error(tool, "PRODUCT_OR_SKU_REQUIRED")
    if any(x is not None and (not isinstance(x, str) or not x.strip()) for x in (product_id, sku_id)):
        return _error(tool, "ARGUMENT_ERROR")
    product = _row(conn.execute("SELECT * FROM products WHERE snapshot_id = ?", (snapshot_id,)))
    sku = _row(conn.execute("SELECT * FROM skus WHERE snapshot_id = ?", (snapshot_id,)))
    actual_product = (product or {}).get("product_id") or (sku or {}).get("product_id")
    if not (product or sku) or (product_id is not None and product_id != actual_product) or (sku_id is not None and sku_id != (sku or {}).get("sku_id")):
        return _error(tool, "PRODUCT_NOT_FOUND", "no_match")
    data = {k: v for k, v in (product or {}).items() if k not in {"snapshot_id", "extra_json"}}
    if sku:
        data.update({k: v for k, v in sku.items() if k not in {"snapshot_id", "extra_json", "stock_status"}})
        data["stock_status"] = sku["stock_status"]
    data["product_extra"] = json.loads((product or {}).get("extra_json", "{}"))
    data["sku_extra"] = json.loads((sku or {}).get("extra_json", "{}"))
    fields = ("product_id", "product_name", "sku_id", "material", "spec", "size", "price_text", "stock_status")
    missing = [k for k in fields if data.get(k) in (None, "", "unknown")]
    refs = [{"source": table, "snapshot_id": snapshot_id} for table, row in (("products", product), ("skus", sku)) if row]
    return ToolResult(True, tool, "partial" if missing else "found", data, missing, refs)
