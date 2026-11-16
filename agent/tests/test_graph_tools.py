"""The graph and native path share one tool catalog and dispatcher."""

import json
import sqlite3
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.graph import build_agent
from agent.handoff import SimulatedHandoffClient
from agent.nodes.handoff import make_handoff_node
from agent.nodes.tools import make_tool_node
from agent.session_state import SessionState
from agent.tool_calling import (
    available_tool_observations, dispatch_graph_call, dispatch_native_call,
    tool_definitions,
)
from agent.tools.common import configure_connection


ROOT = Path(__file__).resolve().parents[2]


class GraphToolTests(unittest.TestCase):
    def setUp(self):
        self.conn = configure_connection(sqlite3.connect(
            ROOT / "data/processed/customer_service/customer_service_train.sqlite"))
        snapshot_id, self.order_id = self.conn.execute(
            "SELECT snapshot_id, order_id FROM orders LIMIT 1").fetchone()
        self.session = SessionState("graph-test", snapshot_id)
        self.state = {"messages": [], "tool_call_count": 0}

    def tearDown(self):
        self.conn.close()

    def call(self, name, args, call_id="call_1"):
        return {"name": name, "args": args, "id": call_id, "type": "tool_call"}

    def test_catalog_is_model_ready_and_single_source(self):
        definitions = tool_definitions()
        self.assertEqual(len(definitions), 10)
        self.assertTrue(all(item["function"]["parameters"]["additionalProperties"] is False
                            for item in definitions))
        names = {item["function"]["name"] for item in definitions}
        self.assertIn("finish_collection", names)
        self.assertIn("get_order_info", names)

    def test_graph_query_records_observation_then_finish(self):
        call = self.call("get_order_info", {"order_id": self.order_id})
        self.state["messages"].append(AIMessage(content="", tool_calls=[call]))
        update = make_tool_node(self.conn, self.session)(self.state)
        self.assertEqual(update["tool_call_count"], 1)
        result = json.loads(update["messages"][0].content)
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["evidence_index"], 0)
        self.state["messages"].extend(update["messages"])
        finish = dispatch_graph_call(
            self.call("finish_collection", {"evidence_indices": [0]}),
            self.state, self.session, self.conn)
        self.assertEqual(finish["route"], "respond")

    def test_finish_indices_only_refer_to_graph_observations(self):
        recent = dispatch_graph_call(
            self.call("get_recent_orders", {}), self.state, self.session, self.conn)
        self.assertTrue(recent["ok"])
        self.state["messages"].append(ToolMessage(
            content=json.dumps(recent), name="get_recent_orders", tool_call_id="recent"))
        order = dispatch_graph_call(
            self.call("get_order_info", {"order_id": self.order_id}),
            self.state, self.session, self.conn)
        self.assertEqual(len(self.session.evidence), 2)
        self.assertEqual(order["evidence_index"], 0)
        self.state["messages"].append(ToolMessage(
            content=json.dumps(order), name="get_order_info", tool_call_id="order"))
        finish = self.call("finish_collection", {"evidence_indices": [0]})
        self.assertEqual(dispatch_graph_call(
            finish, self.state, self.session, self.conn)["route"], "respond")
        self.assertEqual(dispatch_native_call(
            finish, self.session, self.conn)["error_code"], "GRAPH_MESSAGES_REQUIRED")

    def test_invalid_arguments_are_rejected_before_query(self):
        for name, args in (
            ("get_order_info", {"order_id": self.order_id, "snapshot_id": "other"}),
            ("get_product_info", {}),
            ("search_policies", {"query": "发货", "limit": 0}),
            ("handoff_to_human", {"reason_code": "unknown", "reason_detail": "test"}),
            ("finish_collection", {"evidence_indices": [True]}),
        ):
            result = dispatch_graph_call(self.call(name, args), self.state, self.session, self.conn)
            self.assertFalse(result["ok"], (name, args))
        self.assertEqual(self.session.tool_history, [])

    def test_observations_reject_missing_refs_and_foreign_manual_scope(self):
        order = ToolMessage(
            content=json.dumps({
                "tool": "get_order_info", "ok": True, "status": "found",
                "data": {"order_id": self.order_id}, "evidence_refs": [],
            }), name="get_order_info", tool_call_id="missing_refs")
        manual = ToolMessage(
            content=json.dumps({
                "tool": "search_product_manual", "ok": True, "status": "found",
                "data": {"snapshot_id": "OTHER", "product_id": "P", "sku_id": "S",
                         "manual_id": "M"},
                "evidence_refs": [{"source": "frozen_manual_v1", "manual_id": "M"}],
            }), name="search_product_manual", tool_call_id="foreign_manual")
        self.session.current_product_id = "P"
        self.session.current_sku_id = "S"
        self.assertEqual(available_tool_observations(
            {"messages": [order, manual]}, self.session), [])

    def test_parallel_calls_are_rejected_without_execution(self):
        calls = [self.call("get_order_info", {"order_id": self.order_id}, "one"),
                 self.call("request_fallback", {"reason_code": "test"}, "two")]
        self.state["messages"].append(AIMessage(content="", tool_calls=calls))
        update = make_tool_node(self.conn, self.session)(self.state)
        self.assertEqual(len(update["messages"]), 2)
        self.assertEqual(self.session.tool_history, [])

    def test_handoff_reaches_handoff_node(self):
        call = self.call("handoff_to_human", {
            "reason_code": "refund_dispute",
            "reason_detail": "退款尚未到账，用户要求人工核实",
        })
        self.state["messages"].append(AIMessage(content="", tool_calls=[call]))
        update = make_tool_node(self.conn, self.session)(self.state)
        self.state["messages"].extend(update["messages"])
        result = make_handoff_node(SimulatedHandoffClient(), self.session)(self.state)
        self.assertEqual(result["service_status"], "transferred")
        self.assertTrue(result["handoff_result"]["ticket_id"].startswith("SIM-"))

    def test_full_graph_reaches_llmb_after_valid_finish(self):
        class ScriptedModel:
            def __init__(self, responses):
                self.responses = iter(responses)
                self.inputs = []

            def bind_tools(self, definitions):
                self_definitions = {item["function"]["name"] for item in definitions}
                assert "finish_collection" in self_definitions
                return self

            def invoke(self, messages):
                self.inputs.append(messages)
                return next(self.responses)

        llma = ScriptedModel([
            AIMessage(content="", tool_calls=[self.call(
                "get_order_info", {"order_id": self.order_id}, "order")]),
            AIMessage(content="", tool_calls=[self.call(
                "finish_collection", {"evidence_indices": [0]}, "finish")]),
        ])
        llmb = ScriptedModel([AIMessage(content="订单已查询。")])
        graph = build_agent(llma, llmb, self.conn, self.session, SimulatedHandoffClient())
        result = graph.invoke({
            "messages": [HumanMessage(content="我的订单发货了吗？")],
            "tool_call_count": 0,
        })
        self.assertEqual(result["tool_call_count"], 2)
        self.assertEqual(result["messages"][-1].content, "订单已查询。")
        selected = json.loads(llmb.inputs[0][-1].content.split("\n", 1)[1])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["data"]["order_id"], self.order_id)

if __name__ == "__main__":
    unittest.main()
