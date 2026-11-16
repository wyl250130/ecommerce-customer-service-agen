import json
import sqlite3
import unittest
from pathlib import Path

from agent.schemas import ToolAction
from agent.session_state import SessionState
from agent.tool_calling import dispatch_native_call, tool_definitions
from agent.tool_registry import ALLOWED_TOOL_NAMES, execute_tool
from agent.tools.common import configure_connection

ROOT = Path(__file__).resolve().parents[2]


class NativeToolCallingTests(unittest.TestCase):
    def setUp(self):
        self.conn = configure_connection(sqlite3.connect(
            ROOT / "data/processed/customer_service/customer_service_train.sqlite"))
        sid, self.oid = self.conn.execute("SELECT snapshot_id,order_id FROM orders LIMIT 1").fetchone()
        self.state = SessionState("native-test", sid)

    def tearDown(self):
        self.conn.close()

    def test_definitions_match_registered_business_tools(self):
        names = {d["function"]["name"] for d in tool_definitions()}
        self.assertEqual(names, set(ALLOWED_TOOL_NAMES) | {"finish_collection", "request_fallback", "handoff_to_human"})
        self.assertEqual(len(names), 10)
        self.assertTrue(all(d["function"]["parameters"]["additionalProperties"] is False for d in tool_definitions()))

    def test_native_query_records_actual_observation_but_cannot_finish(self):
        result = dispatch_native_call({"name": "get_order_info", "args": {"order_id": self.oid}, "id": "call_1"}, self.state, self.conn)
        self.assertTrue(result["ok"])
        self.assertEqual(len(self.state.evidence), 1)
        signal = dispatch_native_call({"name": "finish_collection", "args": {"evidence_indices": [0]}}, self.state, self.conn)
        self.assertEqual(signal, {"ok": False, "error_code": "GRAPH_MESSAGES_REQUIRED"})
        self.assertNotIn("route", result)

    def test_invalid_terminal_and_scope_calls_are_rejected(self):
        for args in ({"evidence_indices": [0]}, {"evidence_indices": [-1]}, {"evidence_indices": [True]}, {"evidence_indices": []}):
            self.assertFalse(dispatch_native_call({"name": "finish_collection", "args": args}, self.state, self.conn)["ok"])
        result = dispatch_native_call({"name": "get_recent_orders", "args": {"visible_snapshot_ids": ["other"]}}, self.state, self.conn)
        self.assertEqual(result["status"], "forbidden")
        self.assertFalse(dispatch_native_call({"name": "request_fallback", "args": {"reason_code": ""}}, self.state, self.conn)["ok"])

    def test_control_signals_do_not_claim_external_execution(self):
        result = dispatch_native_call({"name": "handoff_to_human", "args": {
            "reason_code": "refund_dispute", "reason_detail": "退款尚未到账，用户要求人工核实",
        }}, self.state, self.conn)
        self.assertEqual(result, {"ok": True, "route": "handoff", "reason_code": "refund_dispute",
                                  "reason_detail": "退款尚未到账，用户要求人工核实"})
        self.assertFalse(dispatch_native_call({"name": "handoff_to_human", "args": {
            "reason_code": "refund_dispute"}}, self.state, self.conn)["ok"])
        self.assertFalse(dispatch_native_call({"name": "handoff_to_human", "args": {
            "reason_code": "made_up", "reason_detail": "test"}}, self.state, self.conn)["ok"])
        self.assertEqual(self.state.tool_history, [])

    def test_policy_adapter_only_returns_approved_source_records(self):
        result = execute_tool(ToolAction("search_policies", {"query": "物流一直没更新怎么办"}), self.state, self.conn)
        self.assertTrue(result.ok)
        for record in result.data["results"]:
            self.assertNotEqual(record["review_status"], "pending")
            raw = json.loads(self.conn.execute("SELECT raw_rule_json FROM rule_documents WHERE policy_id=?", (record["policy_id"],)).fetchone()[0])
            self.assertEqual(record["evidence_required"], raw.get("evidence_required"))
        self.assertEqual(execute_tool(ToolAction("search_policies", {"query": "", "limit": 5}), self.state, self.conn).status, "tool_error")
