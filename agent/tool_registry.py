"""Guarded business-tool dispatch with explicit scope and state updates."""

from __future__ import annotations

import sqlite3
from typing import Any, Callable

from .schemas import ToolAction, ToolResult
from .tools.business import get_after_sales, get_product_info, get_recent_orders
from .tools.logistics import get_logistics
from .tools.manual import ManualSearchService
from .tools.order import get_order_info
from .tools.policies import search_policies


ToolCallable = Callable[..., ToolResult]

TOOL_REGISTRY: dict[str, ToolCallable] = {
    "get_order_info": get_order_info,
    "get_logistics": get_logistics,
    "get_recent_orders": get_recent_orders,
    "get_product_info": get_product_info,
    "get_after_sales": get_after_sales,
    "search_policies": search_policies,
}
MANUAL_TOOL_NAME = "search_product_manual"
ALLOWED_TOOL_NAMES = frozenset(TOOL_REGISTRY) | {MANUAL_TOOL_NAME}
ORDER_TOOLS = frozenset({"get_order_info", "get_logistics", "get_after_sales"})
MANUAL_FORBIDDEN_ARGUMENTS = frozenset({"snapshot_id", "product_id", "sku_id", "manual_id", "top_k"})


def forbidden_scope_arguments(action: ToolAction) -> set[str]:
    forbidden = (MANUAL_FORBIDDEN_ARGUMENTS if action.tool == MANUAL_TOOL_NAME else {"snapshot_id"}) | {
        "visible_snapshot_ids", "user_id", "conn", "state"
    }
    return set(action.arguments).intersection(forbidden)


def execute_tool(
    action: ToolAction,
    *legacy_args: Any,
    conn: sqlite3.Connection | None = None,
    active_snapshot_id: str | None = None,
    default_snapshot_id: str | None = None,
    visible_snapshot_ids: tuple[str, ...] | None = None,
    current_product_id: str | None = None,
    current_sku_id: str | None = None,
    manual_service: ManualSearchService | None = None,
) -> tuple[ToolResult, dict[str, Any]] | ToolResult:
    """Execute with explicit scope; positional legacy calls return ToolResult only.

    The compatibility branch is for the local runner and old graph path. New
    adapters pass keyword fields and receive ``(result, state_updates)``.
    """
    if legacy_args:
        if len(legacy_args) not in {2, 3} or conn is not None:
            raise TypeError("Legacy call expects (action, session, conn[, manual_service])")
        session, legacy_conn = legacy_args[:2]
        if len(legacy_args) == 3:
            if manual_service is not None:
                raise TypeError("manual_service supplied twice")
            manual_service = legacy_args[2]
        result, updates = _execute_scoped(
            action,
            conn=legacy_conn,
            active_snapshot_id=session.snapshot_id,
            default_snapshot_id=session.snapshot_id,
            visible_snapshot_ids=session.visible_snapshot_ids,
            current_product_id=session.current_product_id,
            current_sku_id=session.current_sku_id,
            manual_service=manual_service,
        )
        if result.ok:
            if ("active_snapshot_id" in updates
                    and updates["active_snapshot_id"] != session.snapshot_id):
                # The older runner still reads its Evidence list directly.
                session.evidence.clear()
            for key, value in updates.items():
                setattr(session, "snapshot_id" if key == "active_snapshot_id" else key, value)
        return result

    if conn is None or default_snapshot_id is None:
        raise TypeError("conn and default_snapshot_id are required for explicit tool calls")
    return _execute_scoped(
        action,
        conn=conn,
        active_snapshot_id=active_snapshot_id,
        default_snapshot_id=default_snapshot_id,
        visible_snapshot_ids=visible_snapshot_ids,
        current_product_id=current_product_id,
        current_sku_id=current_sku_id,
        manual_service=manual_service,
    )


def _execute_scoped(
    action: ToolAction, *, conn: sqlite3.Connection,
    active_snapshot_id: str | None, default_snapshot_id: str,
    visible_snapshot_ids: tuple[str, ...] | list[str] | None,
    current_product_id: str | None, current_sku_id: str | None,
    manual_service: ManualSearchService | None,
) -> tuple[ToolResult, dict[str, Any]]:
    """Pure dispatcher: never mutates graph state or a session object."""
    name = action.tool
    if name not in ALLOWED_TOOL_NAMES:
        return ToolResult(
            False, name, "tool_error", error_code="UNKNOWN_TOOL",
            error_message=f"Unknown tool: {name}",
        ), {}

    supplied = forbidden_scope_arguments(action)
    if supplied:
        return ToolResult(
            False, name, "forbidden",
            error_code="SNAPSHOT_ID_FORBIDDEN" if supplied == {"snapshot_id"} else "SCOPE_ARGUMENT_FORBIDDEN",
            error_message=f"Scope/configuration arguments are set by the application: {sorted(supplied)}",
        ), {}

    snapshot_id = active_snapshot_id or default_snapshot_id
    if name == MANUAL_TOOL_NAME:
        if set(action.arguments) != {"query"}:
            return ToolResult(False, name, "tool_error", error_code="ARGUMENT_ERROR",
                              error_message="search_product_manual accepts only a query argument."), {}
        if manual_service is None:
            return ToolResult(False, name, "tool_error", error_code="MANUAL_SERVICE_UNAVAILABLE",
                              error_message="Manual search service is not configured."), {}
        return manual_service.search(
            snapshot_id=snapshot_id, product_id=current_product_id,
            sku_id=current_sku_id, query=action.arguments["query"],
        ), {}

    try:
        if name == "get_recent_orders":
            return get_recent_orders(
                conn=conn, snapshot_id=snapshot_id,
                visible_snapshot_ids=visible_snapshot_ids, **action.arguments,
            ), {}

        selected = snapshot_id
        if name in ORDER_TOOLS and visible_snapshot_ids is not None:
            matches = [sid for sid in visible_snapshot_ids if conn.execute(
                "SELECT 1 FROM orders WHERE snapshot_id=? AND order_id=?",
                (sid, action.arguments.get("order_id")),
            ).fetchone()]
            if len(matches) != 1:
                return ToolResult(
                    False, name, "conflict" if matches else "no_match",
                    error_code="ORDER_SCOPE_AMBIGUOUS" if matches else "ORDER_NOT_FOUND",
                ), {}
            selected = matches[0]

        result = TOOL_REGISTRY[name](conn=conn, snapshot_id=selected, **action.arguments)
        if not result.ok:
            return result, {}
        if name in ORDER_TOOLS:
            product = conn.execute(
                "SELECT product_id FROM products WHERE snapshot_id=?", (selected,)
            ).fetchone()
            sku = conn.execute(
                "SELECT sku_id FROM skus WHERE snapshot_id=?", (selected,)
            ).fetchone()
            return result, {
                "active_snapshot_id": selected,
                "current_order_id": action.arguments["order_id"],
                "current_product_id": product[0] if product else None,
                "current_sku_id": sku[0] if sku else None,
            }
        if name == "get_product_info":
            return result, {
                "current_product_id": result.data.get("product_id") or current_product_id,
                "current_sku_id": result.data.get("sku_id") or current_sku_id,
            }
        return result, {}
    except TypeError as exc:
        return ToolResult(False, name, "tool_error", error_code="ARGUMENT_ERROR",
                          error_message=str(exc)), {}
    except sqlite3.Error as exc:
        return ToolResult(False, name, "tool_error", error_code="SQLITE_ERROR",
                          error_message=str(exc)), {}
