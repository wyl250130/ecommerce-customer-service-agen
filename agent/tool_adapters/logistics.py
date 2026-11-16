"""LangChain adapter for the existing logistics query."""

from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

from ..context import RuntimeContext
from .common import run_business


@tool
def get_logistics(order_id: str, runtime: ToolRuntime[RuntimeContext]) -> Command:
    """查询指定订单的运单、承运商、物流和预计送达信息。"""
    return run_business("get_logistics", {"order_id": order_id}, runtime)
