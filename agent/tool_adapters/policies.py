"""LangChain adapter for reviewed policy search."""

from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

from ..context import RuntimeContext
from .common import run_business


@tool
def search_policies(query: str, runtime: ToolRuntime[RuntimeContext], limit: int = 5) -> Command:
    """检索审核允许的规则卡及适用条件、例外和来源。"""
    return run_business("search_policies", {"query": query, "limit": limit}, runtime)
