"""One tool catalog and dispatcher for native calls and the LangGraph flow.

The catalog is passed directly to the model's ``bind_tools`` method. The
graph's tool node calls ``dispatch_graph_call`` and wraps its result in a
ToolMessage. This module remains importable without LangChain installed.
"""
import copy
import json
from typing import Any

from .handoff import HandoffReason
from .schemas import ToolAction
from .session_state import SessionState
from .tool_registry import execute_tool


def _definition(name, description, properties, required=()):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": list(required), "additionalProperties": False}}}


STRING = {"type": "string", "minLength": 1}
LIMIT = {"type": "integer", "minimum": 1, "maximum": 20, "default": 5}
BUSINESS_DEFINITIONS = [
    _definition("get_recent_orders", "按需查询会话范围内订单候选；无真实用户归属或下单时间，不保证按最近时间排序。", {"limit": LIMIT, "product_hint": STRING}),
    _definition("get_order_info", "查询已确定订单的付款、发货、商品和地址处理状态。", {"order_id": STRING}, ["order_id"]),
    _definition("get_logistics", "查询已确定订单的运单、承运商、物流和预计送达；缺失值不能视为承诺。", {"order_id": STRING}, ["order_id"]),
    _definition("get_product_info", "查询当前场景商品/SKU属性、价格文本与库存；product_id或sku_id至少一个。", {"product_id": STRING, "sku_id": STRING}),
    _definition("get_after_sales", "查询指定订单退货、退款、补发、补偿状态；不执行任何业务写操作。", {"order_id": STRING}, ["order_id"]),
    _definition("search_policies", "检索审核允许的规则卡，返回条件、例外、证据要求和来源；需自行判断适用性。", {"query": STRING, "limit": LIMIT}, ["query"]),
    _definition("search_product_manual", "检索会话已精确绑定商品/SKU的说明书章节；结果标明商品专属或类别级来源，found不代表问题有答案。", {"query": STRING}, ["query"]),
]
CONTROL_DEFINITIONS = [
    _definition("finish_collection", "证据足以交给LLM B时调用；evidence_indices引用当前消息中可用业务工具结果的零起始索引。不是直接回复用户。",
                {"evidence_indices": {"type": "array", "items": {"type": "integer", "minimum": 0}, "minItems": 1}}, ["evidence_indices"]),
    _definition("request_fallback", "必要证据无法再获取时请求应用脚本兜底，不调用LLM B。", {"reason_code": STRING}, ["reason_code"]),
    _definition("handoff_to_human", "请求转人工，并说明具体原因；仅路由信号，实际接通由Handoff Node确认。",
                {"reason_code": {"type": "string", "enum": [reason.value for reason in HandoffReason]}, "reason_detail": STRING},
                ["reason_code", "reason_detail"]),
]


def tool_definitions():
    """Return the single JSON schema catalog used for model binding."""
    return copy.deepcopy(BUSINESS_DEFINITIONS + CONTROL_DEFINITIONS)


BUSINESS_NAMES = {item["function"]["name"] for item in BUSINESS_DEFINITIONS}
TOOL_SCHEMAS = {
    item["function"]["name"]: item["function"]["parameters"]
    for item in BUSINESS_DEFINITIONS + CONTROL_DEFINITIONS
}


def _valid_arguments(schema: dict, value: Any) -> bool:
    """Validate the small JSON-schema subset used by this tool catalog."""
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            return False
        properties = schema.get("properties", {})
        if any(key not in value for key in schema.get("required", [])):
            return False
        if schema.get("additionalProperties") is False and any(
            key not in properties for key in value
        ):
            return False
        return all(_valid_arguments(properties[key], item) for key, item in value.items())
    if kind == "string":
        return (isinstance(value, str)
                and len(value) >= schema.get("minLength", 0)
                and ("enum" not in schema or value in schema["enum"]))
    if kind == "integer":
        return (type(value) is int
                and value >= schema.get("minimum", value)
                and value <= schema.get("maximum", value))
    if kind == "array":
        return (isinstance(value, list)
                and len(value) >= schema.get("minItems", 0)
                and all(_valid_arguments(schema["items"], item) for item in value))
    raise ValueError(f"Unsupported tool schema type: {kind}")


def available_tool_observations(graph_state: dict, session_context) -> list[dict[str, Any]]:
    """Usable business observations in graph messages, in time order."""
    from langchain_core.messages import ToolMessage

    observations = []
    for message in graph_state.get("messages", []):
        if (not isinstance(message, ToolMessage) or message.name not in BUSINESS_NAMES
                or message.name == "get_recent_orders"):
            continue
        try:
            result = json.loads(message.content)
        except (TypeError, ValueError):
            continue
        if (not isinstance(result, dict) or result.get("tool") != message.name
                or result.get("ok") is not True
                or result.get("status") not in {"found", "partial"}):
            continue
        data = result.get("data")
        refs = result.get("evidence_refs")
        if (not isinstance(data, dict) or not data or not isinstance(refs, list) or not refs
                or any(not isinstance(ref, dict) for ref in refs)):
            continue
        if message.name == "search_policies":
            if any(ref.get("source") != "rule_documents" or not ref.get("policy_id") for ref in refs):
                continue
        elif message.name == "search_product_manual":
            if (not data.get("manual_id")
                    or not session_context.current_product_id
                    or not session_context.current_sku_id
                    or data.get("snapshot_id") != session_context.snapshot_id
                    or data.get("product_id") != session_context.current_product_id
                    or data.get("sku_id") != session_context.current_sku_id
                    or any(ref.get("source") not in {"frozen_manual_v1", "legacy_category_manual"}
                           or ref.get("manual_id") != data.get("manual_id")
                           or (ref.get("source") == "legacy_category_manual" and
                               (ref.get("snapshot_id") != session_context.snapshot_id or
                                ref.get("product_id") != session_context.current_product_id or
                                ref.get("sku_id") != session_context.current_sku_id))
                           for ref in refs)):
                continue
        elif any(ref.get("snapshot_id") != session_context.snapshot_id for ref in refs):
            continue
        observations.append(result)
    return observations


def dispatch_graph_call(call, graph_state, session_context, conn, manual_service=None):
    """Validate a graph tool call against the catalog and execute it once."""
    from langchain_core.messages import ToolMessage

    if not isinstance(call, dict) or call.get("name") not in TOOL_SCHEMAS:
        return {"ok": False, "error_code": "UNKNOWN_TOOL"}
    name, args = call["name"], call.get("args", {})
    if not _valid_arguments(TOOL_SCHEMAS[name], args):
        return {"ok": False, "error_code": "ARGUMENT_ERROR"}
    if name == "get_product_info" and not (args.get("product_id") or args.get("sku_id")):
        return {"ok": False, "error_code": "ARGUMENT_ERROR"}

    if name == "finish_collection":
        available = available_tool_observations(graph_state, session_context)
        indices = args["evidence_indices"]
        if any(index >= len(available) for index in indices):
            return {"ok": False, "error_code": "INVALID_EVIDENCE_REFERENCES"}
        return {"ok": True, "route": "respond", "evidence_indices": indices}

    before = available_tool_observations(graph_state, session_context) if name in BUSINESS_NAMES else []
    result = dispatch_native_call(call, session_context, conn, manual_service)
    if name in BUSINESS_NAMES and result.get("ok"):
        candidate = ToolMessage(
            content=json.dumps(result, ensure_ascii=False),
            name=name,
            tool_call_id=call.get("id") or "pending",
        )
        probe = {**graph_state, "messages": [*graph_state.get("messages", []), candidate]}
        if len(available_tool_observations(probe, session_context)) > len(before):
            result["evidence_index"] = len(before)
    return result


def dispatch_native_call(call, session_context: SessionState, conn, manual_service=None):
    """Execute one call using the trusted SessionState, not LangGraph State.

    The caller owns sequencing and ToolMessages. ``finish_collection`` needs
    graph messages and must go through ``dispatch_graph_call`` instead.

    Terminal signals must be handled by a graph conditional edge, not blindly
    looped back into the planner. This does not infer semantic sufficiency.
    """
    if not isinstance(call, dict):
        return {"ok": False, "error_code": "INVALID_TOOL_CALL"}
    name, args = call.get("name"), call.get("args", {})
    if not isinstance(args, dict):
        return {"ok": False, "error_code": "ARGUMENT_ERROR"}
    if name == "finish_collection":
        return {"ok": False, "error_code": "GRAPH_MESSAGES_REQUIRED"}
    if name == "handoff_to_human":
        if set(args) != {"reason_code", "reason_detail"}:
            return {"ok": False, "error_code": "ARGUMENT_ERROR"}
        try:
            reason_code = HandoffReason(args["reason_code"])
        except (ValueError, TypeError):
            return {"ok": False, "error_code": "INVALID_HANDOFF_REASON"}
        detail = args["reason_detail"]
        if not isinstance(detail, str) or not detail.strip():
            return {"ok": False, "error_code": "ARGUMENT_ERROR"}
        return {"ok": True, "route": "handoff", "reason_code": reason_code.value, "reason_detail": detail.strip()}
    if name == "request_fallback":
        if set(args) != {"reason_code"} or not isinstance(args["reason_code"], str) or not args["reason_code"].strip():
            return {"ok": False, "error_code": "ARGUMENT_ERROR"}
        return {"ok": True, "route": "fallback", "reason_code": args["reason_code"]}
    action = ToolAction(name, args)
    result = execute_tool(action, session_context, conn, manual_service)
    session_context.add_tool_result(action, result)
    return result.to_dict()
