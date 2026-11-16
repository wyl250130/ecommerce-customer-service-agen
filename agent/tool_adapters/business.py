"""Adapters for order candidates, product data, and after-sales status."""

from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

from ..context import RuntimeContext
from .common import run_business


@tool
def get_recent_orders(
    runtime: ToolRuntime[RuntimeContext], limit: int = 5, product_hint: str | None = None,
) -> Command:
    """查询会话可见范围内的订单候选；结果不保证按下单时间排序。"""
    args = {"limit": limit}
    if product_hint:
        args["product_hint"] = product_hint
    return run_business("get_recent_orders", args, runtime)


@tool
def get_product_info(
    runtime: ToolRuntime[RuntimeContext], product_id: str | None = None,
    sku_id: str | None = None,
) -> Command:
    """查询当前场景的商品或 SKU 信息；至少提供一个 ID。"""
    args = {}
    if product_id:
        args["product_id"] = product_id
    if sku_id:
        args["sku_id"] = sku_id
    return run_business("get_product_info", args, runtime)


@tool
def get_after_sales(order_id: str, runtime: ToolRuntime[RuntimeContext]) -> Command:
    """查询指定订单的退货、退款、补发或补偿状态。"""
    return run_business("get_after_sales", {"order_id": order_id}, runtime)
