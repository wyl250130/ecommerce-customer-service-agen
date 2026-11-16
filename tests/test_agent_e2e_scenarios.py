import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from agent.tools.common import configure_connection
from agent.tools.orders.info import get_order_info
from agent.tools.products.info import get_product_info
from agent.tools.manuals.legacy import LegacyManualSearchService
from agent.tool_registry import execute_tool
from agent.schemas import ToolAction
from rag.unified_manuals import UnifiedManualCorpus
from scripts.build_agent_e2e_scenarios import DEMO_DB, TRAIN_DB, build


class AgentE2EScenarioTests(unittest.TestCase):
    def test_all_scenarios_keep_order_product_and_manual_in_one_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = build(Path(folder))
            self.assertEqual(manifest["counts"]["train"], 51)
            self.assertEqual(manifest["counts"]["demo"], 1)
            self.assertEqual(manifest["counts"]["distinct_legacy_manuals"], 8)
            corpus = UnifiedManualCorpus()
            with (Path(folder) / "scenarios.jsonl").open(encoding="utf-8") as stream:
                cases = [json.loads(line) for line in stream]
            for case in cases:
                with self.subTest(case=case["scenario_id"]):
                    db_path = DEMO_DB if case["split"] == "demo" else TRAIN_DB
                    with closing(configure_connection(sqlite3.connect(db_path))) as db:
                        order = get_order_info(db, case["snapshot_id"], case["order_id"])
                        product = get_product_info(db, case["snapshot_id"],
                                                   case["product_id"], case["sku_id"])
                        manual, updates = execute_tool(
                            ToolAction("search_product_manual", {"query": "使用步骤 怎么使用"}),
                            conn=db, active_snapshot_id=case["snapshot_id"],
                            default_snapshot_id=case["snapshot_id"],
                            visible_snapshot_ids=(case["snapshot_id"],),
                            current_product_id=case["product_id"],
                            current_sku_id=case["sku_id"],
                            manual_service=LegacyManualSearchService(db_path))
                    self.assertTrue(order.ok)
                    self.assertTrue(product.ok)
                    self.assertTrue(manual.ok)
                    self.assertEqual(updates, {})
                    self.assertEqual(manual.data["manual_id"], case["manual_id"])
                    self.assertEqual(manual.data["source_tier"], "legacy_category")
                    self.assertTrue(any(row["chunk_id"] == case["manual_chunk_id"]
                                        for row in manual.data["results"]))
                    self.assertEqual(order.data["product_id"], case["product_id"])
                    self.assertEqual(order.data["sku_id"], case["sku_id"])
                    self.assertEqual(product.data["product_name"], case["product_name"])
                    self.assertEqual(order.data["shipment_status"], case["shipment_status"])
                    chunks = corpus.candidates(case["snapshot_id"], case["product_id"], case["sku_id"])
                    selected = [row for row in chunks if row["chunk_id"] == case["manual_chunk_id"]]
                    self.assertEqual(len(selected), 1)
                    self.assertEqual(selected[0]["manual_id"], case["manual_id"])
                    self.assertEqual(selected[0]["content"], case["manual_chunk_content"])
                    self.assertIn(case["order_id"], case["reference_response"])
                    self.assertIn(case["product_name"], case["reference_response"])
                    self.assertIn(case["manual_chunk_content"], case["reference_response"])
                    self.assertEqual(corpus.candidates("wrong-snapshot", case["product_id"],
                                                       case["sku_id"]), [])


if __name__ == "__main__":
    unittest.main()
