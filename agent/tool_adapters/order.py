"""LangChain adapter for the existing order query."""

from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

from ..context import RuntimeContext
from .common import run_business


@tool
def get_order_info(order_id: str, runtime: ToolRuntime[RuntimeContext]) -> Command:
    """查询指定订单的付款、发货、商品和地址处理状态。"""
    return run_business("get_order_info", {"order_id": order_id}, runtime)
