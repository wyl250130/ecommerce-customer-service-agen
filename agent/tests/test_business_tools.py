import json
import sqlite3
import unittest
from pathlib import Path

from agent.schemas import ToolAction
from agent.session_state import SessionState
from agent.tool_registry import execute_tool
from agent.tools.common import configure_connection

ROOT = Path(__file__).resolve().parents[2]


class BusinessTests(unittest.TestCase):
    def setUp(self):
        self.conn = configure_connection(sqlite3.connect(
            ROOT / "data/processed/customer_service/customer_service_train.sqlite"))
        self.sid, self.oid = self.conn.execute("SELECT snapshot_id,order_id FROM orders LIMIT 1").fetchone()
        self.state = SessionState("test", self.sid)

    def tearDown(self):
        self.conn.close()

    def call(self, tool, **arguments):
        return execute_tool(ToolAction(tool, arguments), self.state, self.conn)

    def test_recent_scope_filter_and_no_invented_time(self):
        ids = [r[0] for r in self.conn.execute("SELECT snapshot_id FROM orders LIMIT 2")]
        self.state.visible_snapshot_ids = ids
        result = self.call("get_recent_orders")
        self.assertEqual(result.data["total"], 2)
        self.assertTrue(all(x["order_time"] is None for x in result.data["orders"]))
        self.assertFalse(result.data["ownership_verified"])
        self.assertEqual({x["snapshot_id"] for x in result.evidence_refs}, set(ids))
        self.assertEqual(self.call("get_recent_orders", product_hint="NONEXISTENT").status, "no_match")

    def test_scope_cannot_be_supplied_by_model(self):
        for tool in ("get_recent_orders", "get_product_info", "get_after_sales"):
            self.assertEqual(self.call(tool, visible_snapshot_ids=["other"]).status, "forbidden")

    def test_recent_parameters(self):
        for limit in (0, True, 100, "5"):
            self.assertEqual(self.call("get_recent_orders", limit=limit).status, "tool_error")

    def test_product_and_after_sales_preserve_source(self):
        sid, pid, sku = self.conn.execute("SELECT p.snapshot_id,p.product_id,s.sku_id FROM products p JOIN skus s USING(snapshot_id) WHERE p.product_id IS NOT NULL LIMIT 1").fetchone()
        self.state.snapshot_id = sid
        result = self.call("get_product_info", product_id=pid, sku_id=sku)
        raw = json.loads(self.conn.execute("SELECT raw_fact_json FROM fact_snapshots WHERE snapshot_id=?", (sid,)).fetchone()[0])
        self.assertTrue(result.ok)
        self.assertEqual(result.data["product_name"], raw["product_name"])
        self.assertEqual(result.data["stock_status"], raw["stock_status"])
        self.state.add_tool_result(ToolAction("get_product_info", {"product_id": pid, "sku_id": sku}), result)
        self.assertEqual(self.state.current_sku_id, sku)
        self.assertEqual(self.call("get_product_info", product_id="WRONG").status, "no_match")
        self.state.snapshot_id = self.sid
        result = self.call("get_after_sales", order_id=self.oid)
        self.assertTrue(result.ok)
        self.assertEqual(self.call("get_after_sales", order_id="WRONG").status, "no_match")

    def test_select_candidate_and_reject_foreign_order(self):
        other = self.conn.execute("SELECT snapshot_id,order_id FROM orders WHERE snapshot_id != ? LIMIT 1", (self.sid,)).fetchone()
        self.state.visible_snapshot_ids = [self.sid, other[0]]
        result = self.call("get_order_info", order_id=other[1])
        self.assertTrue(result.ok)
        self.assertEqual(self.state.snapshot_id, other[0])
        self.state.visible_snapshot_ids = [self.sid]
        self.assertEqual(self.call("get_after_sales", order_id=other[1]).status, "no_match")

    def test_explicit_registry_returns_updates_without_mutating_session(self):
        other = self.conn.execute(
            "SELECT snapshot_id,order_id FROM orders WHERE snapshot_id != ? LIMIT 1",
            (self.sid,),
        ).fetchone()
        result, updates = execute_tool(
            ToolAction("get_order_info", {"order_id": other[1]}),
            conn=self.conn,
            active_snapshot_id=self.sid,
            default_snapshot_id=self.sid,
            visible_snapshot_ids=(self.sid, other[0]),
            current_product_id=None,
            current_sku_id=None,
        )
        self.assertTrue(result.ok)
        self.assertEqual(updates["active_snapshot_id"], other[0])
        self.assertEqual(updates["current_order_id"], other[1])
        self.assertIsNone(self.state.current_order_id)
        self.assertEqual(self.state.snapshot_id, self.sid)
        self.assertEqual(self.state.evidence, [])

        denied, denied_updates = execute_tool(
            ToolAction("get_order_info", {"order_id": other[1]}),
            conn=self.conn,
            active_snapshot_id=self.sid,
            default_snapshot_id=self.sid,
            visible_snapshot_ids=(self.sid,),
            current_product_id=None,
            current_sku_id=None,
        )
        self.assertFalse(denied.ok)
        self.assertEqual(denied.error_code, "ORDER_NOT_FOUND")
        self.assertEqual(denied_updates, {})

    def test_all_imported_facts_equal_latest_train(self):
        if not (ROOT / "data/processed/gold_sft/final_semantic_v3_consistency/train_sft_gold.jsonl").exists():
            self.skipTest("Private training Gold is not included in the public repository")
        with (ROOT / "data/processed/gold_sft/final_semantic_v3_consistency/train_sft_gold.jsonl").open() as handle:
            expected = {r["input"]["fact_context"]["snapshot_id"]: r["input"]["fact_context"] for r in map(json.loads, handle)}
        actual = {r[0]: json.loads(r[1]) for r in self.conn.execute("SELECT snapshot_id,raw_fact_json FROM fact_snapshots")}
        self.assertEqual(actual, expected)
