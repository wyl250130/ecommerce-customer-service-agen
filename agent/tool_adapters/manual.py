"""LangChain adapter for the frozen product manual service."""

from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

from ..context import RuntimeContext
from .common import run_business


@tool
def search_product_manual(query: str, runtime: ToolRuntime[RuntimeContext]) -> Command:
    """检索当前已绑定商品和 SKU 的冻结说明书章节。"""
    return run_business("search_product_manual", {"query": query}, runtime)
