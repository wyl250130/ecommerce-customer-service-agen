"""End-to-end checks for injected ToolRuntime and ToolNode state updates."""

import json
import sqlite3
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.context import RuntimeContext
from agent.evidence import available_tool_observations
from agent.graph import build_agent
from agent.handoff import SimulatedHandoffClient
from agent.nodes.llmb import make_llmb_node
from agent.state import make_initial_state
from agent.tool_adapters import ALL_TOOLS
from agent.tools.common import configure_connection


DB = Path(__file__).resolve().parents[2] / "data/processed/customer_service/customer_service_train.sqlite"


class ScriptedModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.inputs = []

    def bind_tools(self, tools):
        assert {tool.name for tool in tools} == {tool.name for tool in ALL_TOOLS}
        return self

    def invoke(self, messages):
        self.inputs.append(messages)
        return next(self.responses)


class AdapterGraphTests(unittest.TestCase):
    def setUp(self):
        self.conn = configure_connection(sqlite3.connect(DB))
        self.orders = self.conn.execute(
            "SELECT snapshot_id,order_id FROM orders GROUP BY snapshot_id ORDER BY snapshot_id LIMIT 2"
        ).fetchall()

    def tearDown(self):
        self.conn.close()

    @staticmethod
    def call(name, args, call_id):
        return {"name": name, "args": args, "id": call_id, "type": "tool_call"}

    def test_model_schema_hides_runtime(self):
        for tool in ALL_TOOLS:
            self.assertNotIn("runtime", tool.tool_call_schema.model_fields)

    def test_context_rejects_missing_connection_factory(self):
        sid, _ = self.orders[0]
        with self.assertRaisesRegex(TypeError, "connection_factory"):
            RuntimeContext(connection_factory=None, default_snapshot_id=sid)

    def test_graph_rejects_missing_initial_counters_before_model_runs(self):
        sid, _ = self.orders[0]
        graph = build_agent(ScriptedModel([]), ScriptedModel([]))
        context = RuntimeContext(
            connection_factory=lambda: configure_connection(sqlite3.connect(DB)),
            default_snapshot_id=sid,
        )
        with self.assertRaisesRegex(ValueError, "make_initial_state"):
            graph.invoke({"messages": [HumanMessage(content="查询订单")]}, context=context)

    def test_switch_order_expires_old_observation_and_resets_index(self):
        (sid_a, order_a), (sid_b, order_b) = self.orders
        llma = ScriptedModel([
            AIMessage(content="", tool_calls=[self.call("get_order_info", {"order_id": order_a}, "a")]),
            AIMessage(content="", tool_calls=[self.call("get_order_info", {"order_id": order_b}, "b")]),
            AIMessage(content="", tool_calls=[self.call("finish_collection", {"evidence_indices": [0]}, "finish")]),
        ])
        llmb = ScriptedModel([AIMessage(content="查询完成")])
        graph = build_agent(llma, llmb)
        context = RuntimeContext(
            default_snapshot_id=sid_a,
            visible_snapshot_ids=(sid_a, sid_b),
            conversation_id="adapter-test", handoff_client=SimulatedHandoffClient(),
            connection_factory=lambda: configure_connection(sqlite3.connect(DB)),
        )
        initial_state = make_initial_state(
            [HumanMessage(content="查询订单")], context=context)
        state = graph.invoke(initial_state, context=context)
        self.assertEqual(state["messages"][-1].content, "查询完成")
        self.assertEqual(state["active_snapshot_id"], sid_b)
        self.assertEqual(state["current_order_id"], order_b)
        self.assertEqual(state["evidence_generation"], 2)
        self.assertEqual(state["tool_call_count"], 3)
        order_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)
                          and m.name == "get_order_info"]
        self.assertEqual(len(order_messages), 2)
        self.assertEqual([m.artifact["evidence_generation"] for m in order_messages], [1, 2])
        current = available_tool_observations(state["messages"], state)
        self.assertEqual(len(current), 1)
        self.assertEqual(current[0]["data"]["order_id"], order_b)
        self.assertEqual(json.loads(order_messages[-1].content)["evidence_index"], 0)

    def test_first_order_selection_invalidates_prior_policy_evidence(self):
        (sid_a, _), (sid_b, order_b) = self.orders
        llma = ScriptedModel([
            AIMessage(content="", tool_calls=[self.call(
                "search_policies", {"query": "物流"}, "policy")]),
            AIMessage(content="", tool_calls=[self.call(
                "get_order_info", {"order_id": order_b}, "order")]),
            AIMessage(content="已取得新订单资料"),
        ])
        graph = build_agent(llma, ScriptedModel([]))
        context = RuntimeContext(
            connection_factory=lambda: configure_connection(sqlite3.connect(DB)),
            default_snapshot_id=sid_a, visible_snapshot_ids=(sid_a, sid_b),
        )
        state = graph.invoke(make_initial_state(
            [HumanMessage(content="查物流")], context=context), context=context)
        observations = [m for m in state["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual([m.name for m in observations], ["search_policies", "get_order_info"])
        self.assertTrue(json.loads(observations[0].content)["ok"])
        self.assertEqual(state["evidence_generation"], 1)
        self.assertEqual([m.artifact["evidence_generation"] for m in observations], [0, 1])
        current = available_tool_observations(state["messages"], state)
        self.assertEqual([item["tool"] for item in current], ["get_order_info"])
        self.assertEqual(json.loads(observations[1].content)["evidence_index"], 0)

    def test_product_binding_change_increments_evidence_generation(self):
        sid, product_id, sku_id = self.conn.execute(
            "SELECT p.snapshot_id,p.product_id,s.sku_id FROM products p "
            "JOIN skus s USING(snapshot_id) WHERE p.product_id IS NOT NULL LIMIT 1",
        ).fetchone()
        llma = ScriptedModel([
            AIMessage(content="", tool_calls=[self.call(
                "get_product_info", {"product_id": product_id, "sku_id": sku_id}, "product")]),
            AIMessage(content="已查询商品"),
        ])
        graph = build_agent(llma, ScriptedModel([]))
        context = RuntimeContext(
            connection_factory=lambda: configure_connection(sqlite3.connect(DB)),
            default_snapshot_id=sid,
        )
        state = graph.invoke(make_initial_state(
            [HumanMessage(content="查商品")], context=context), context=context)
        self.assertEqual(state["current_product_id"], product_id)
        self.assertEqual(state["current_sku_id"], sku_id)
        self.assertEqual(state["evidence_generation"], 1)
        observation = next(m for m in state["messages"] if isinstance(m, ToolMessage))
        self.assertEqual(observation.artifact["evidence_generation"], 1)
        self.assertEqual(json.loads(observation.content)["evidence_index"], 0)

    def test_out_of_scope_order_does_not_change_current_order(self):
        (sid_a, _), (_, order_b) = self.orders
        llma = ScriptedModel([
            AIMessage(content="", tool_calls=[self.call("get_order_info", {"order_id": order_b}, "foreign")]),
            AIMessage(content="需要核对订单号"),
        ])
        graph = build_agent(llma, ScriptedModel([]))
        context = RuntimeContext(
            default_snapshot_id=sid_a, visible_snapshot_ids=(sid_a,),
            connection_factory=lambda: configure_connection(sqlite3.connect(DB)),
        )
        state = graph.invoke(make_initial_state(
            [HumanMessage(content="查询另一个订单")], context=context), context=context)
        result = next(m for m in state["messages"] if isinstance(m, ToolMessage))
        self.assertFalse(json.loads(result.content)["ok"])
        self.assertIsNone(state["current_order_id"])
        self.assertEqual(state["evidence_generation"], 0)

    def test_parallel_batch_is_rejected_before_any_query(self):
        (sid_a, order_a) = self.orders[0]
        llma = ScriptedModel([
            AIMessage(content="", tool_calls=[
                self.call("get_order_info", {"order_id": order_a}, "one"),
                self.call("request_fallback", {"reason_code": "test"}, "two"),
            ]),
            AIMessage(content="请稍等"),
        ])
        graph = build_agent(llma, ScriptedModel([]))
        context = RuntimeContext(
            connection_factory=lambda: configure_connection(sqlite3.connect(DB)),
            default_snapshot_id=sid_a,
        )
        state = graph.invoke(make_initial_state(
            [HumanMessage(content="查询订单")], context=context), context=context)
        errors = [json.loads(m.content) for m in state["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual(len(errors), 2)
        self.assertTrue(all(x["error_code"] == "MULTIPLE_TOOL_CALLS_UNSUPPORTED" for x in errors))
        self.assertNotIn("tool_history", state)

    def test_finish_without_evidence_returns_to_llma(self):
        sid_a, _ = self.orders[0]
        llma = ScriptedModel([
            AIMessage(content="", tool_calls=[self.call(
                "finish_collection", {"evidence_indices": [0]}, "finish")]),
            AIMessage(content="还需要查询订单"),
        ])
        graph = build_agent(llma, ScriptedModel([]))
        context = RuntimeContext(
            connection_factory=lambda: configure_connection(sqlite3.connect(DB)),
            default_snapshot_id=sid_a,
        )
        state = graph.invoke(make_initial_state(
            [HumanMessage(content="发货了吗")], context=context), context=context)
        result = next(m for m in state["messages"] if isinstance(m, ToolMessage))
        self.assertEqual(json.loads(result.content)["error_code"], "INVALID_EVIDENCE_REFERENCES")
        self.assertEqual(state["messages"][-1].content, "还需要查询订单")

    def test_llmb_receives_only_selected_current_evidence(self):
        sid, order_id = self.orders[0]
        def observation(call_id, status):
            return ToolMessage(
                content=json.dumps({
                    "tool": "get_order_info", "ok": True, "status": "found",
                    "data": {"order_id": order_id, "shipment_status": status},
                    "evidence_refs": [{"snapshot_id": sid}],
                }),
                name="get_order_info", tool_call_id=call_id,
                artifact={"evidence_generation": 0},
            )

        model = ScriptedModel([AIMessage(content="已查询")])
        state = {
            "active_snapshot_id": sid, "evidence_generation": 0,
            "messages": [
                HumanMessage(content="这单发货了吗？"),
                observation("old", "older_value"),
                observation("chosen", "selected_value"),
                ToolMessage(content=json.dumps({
                    "ok": True, "route": "respond", "evidence_indices": [1],
                }), name="finish_collection", tool_call_id="finish"),
            ],
        }
        make_llmb_node(model)(state)
        llmb_input = model.inputs[0]
        self.assertEqual(len(llmb_input), 3)
        self.assertEqual(llmb_input[1].content, "这单发货了吗？")
        selected = json.loads(llmb_input[2].content.split("\n", 1)[1])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["data"]["shipment_status"], "selected_value")

    def test_handoff_uses_context_gateway(self):
        sid_a, _ = self.orders[0]
        llma = ScriptedModel([AIMessage(content="", tool_calls=[self.call(
            "handoff_to_human",
            {"reason_code": "user_requested", "reason_detail": "用户希望人工处理"},
            "handoff")])])
        graph = build_agent(llma, ScriptedModel([]))
        context = RuntimeContext(
            connection_factory=lambda: configure_connection(sqlite3.connect(DB)),
            default_snapshot_id=sid_a,
            conversation_id="adapter-handoff", handoff_client=SimulatedHandoffClient(),
        )
        state = graph.invoke(make_initial_state(
            [HumanMessage(content="转人工")], context=context), context=context)
        self.assertEqual(state["service_status"], "transferred")
        self.assertTrue(state["handoff_result"]["ticket_id"].startswith("SIM-"))


if __name__ == "__main__":
    unittest.main()
