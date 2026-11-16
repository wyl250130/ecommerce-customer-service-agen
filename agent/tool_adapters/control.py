"""Control signals emitted through ToolNode, with graph evidence indices."""

from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

from ..context import RuntimeContext
from ..evidence import available_tool_observations
from ..handoff import HandoffReason
from .common import run_control


@tool
def finish_collection(
    evidence_indices: list[int], runtime: ToolRuntime[RuntimeContext],
) -> Command:
    """确认当前证据已足够交给 LLM B；索引从零开始。"""
    state = runtime.state
    available = available_tool_observations(state.get("messages", []), state)
    if (not evidence_indices or any(type(i) is not int or i < 0 or i >= len(available)
                                    for i in evidence_indices)):
        payload = {"ok": False, "error_code": "INVALID_EVIDENCE_REFERENCES"}
    else:
        payload = {"ok": True, "route": "respond", "evidence_indices": evidence_indices}
    return run_control("finish_collection", payload, runtime)


@tool
def request_fallback(reason_code: str, runtime: ToolRuntime[RuntimeContext]) -> Command:
    """证据无法继续取得时请求应用脚本兜底。"""
    if not reason_code.strip():
        payload = {"ok": False, "error_code": "ARGUMENT_ERROR"}
    else:
        payload = {"ok": True, "route": "fallback", "reason_code": reason_code.strip()}
    return run_control("request_fallback", payload, runtime)


@tool
def handoff_to_human(
    reason_code: str, reason_detail: str, runtime: ToolRuntime[RuntimeContext],
) -> Command:
    """请求转人工；实际接通由后续 handoff 节点确认。"""
    try:
        reason = HandoffReason(reason_code)
    except ValueError:
        payload = {"ok": False, "error_code": "INVALID_HANDOFF_REASON"}
    else:
        payload = ({"ok": True, "route": "handoff", "reason_code": reason.value,
                    "reason_detail": reason_detail.strip()}
                   if reason_detail.strip() else
                   {"ok": False, "error_code": "ARGUMENT_ERROR"})
    return run_control("handoff_to_human", payload, runtime)
