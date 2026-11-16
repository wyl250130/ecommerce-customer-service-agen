"""One coherent synthetic event through the actual Graph and three business tools."""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.context import RuntimeContext
from agent.evidence import available_tool_observations
from agent.graph import build_agent
from agent.state import make_initial_state
from agent.tools.common import configure_connection
from agent.tools.manual import ManualSearchService
from agent.tools.manuals.legacy import LegacyManualSearchService
from scripts.build_agent_full_chain_fixture import SOURCE, build, hash_file


class ScriptedA:
    def __init__(self, order_id, product_id, sku_id, manual_query="如何安全用于耳部清洁？"):
        self.calls = iter([
            ("get_order_info", {"order_id": order_id}),
            ("get_product_info", {"product_id": product_id, "sku_id": sku_id}),
            ("search_product_manual", {"query": manual_query}),
            ("finish_collection", {"evidence_indices": [0, 1, 2]}),
        ])

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        name, args = next(self.calls)
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": name}])


class ScriptedB:
    def __init__(self, reference):
        self.reference = reference
        self.selected = None

    def invoke(self, messages):
        self.selected = messages[-1].content
        return AIMessage(content=self.reference)


class FullChainFixtureTests(unittest.TestCase):
    def test_existing_demo_order_reaches_legacy_manual_through_graph(self):
        root = Path(__file__).resolve().parents[2]
        with (root / "data/processed/agent_e2e_v1/scenarios.jsonl").open(encoding="utf-8") as stream:
            case = next(row for line in stream if (row := json.loads(line))["split"] == "demo")
        db_path = root / "data/processed/customer_service/customer_service_chat_e2e.sqlite"
        context = RuntimeContext(
            connection_factory=lambda: configure_connection(sqlite3.connect(db_path)),
            default_snapshot_id=case["snapshot_id"],
            visible_snapshot_ids=(case["snapshot_id"],),
            manual_service=LegacyManualSearchService(db_path),
        )
        llmb = ScriptedB(case["reference_response"])
        graph = build_agent(ScriptedA(case["order_id"], case["product_id"], case["sku_id"],
                                      manual_query="使用步骤 怎么使用"), llmb)
        state = graph.invoke(make_initial_state(
            [HumanMessage(content=case["user_query"])], context=context), context=context)
        observations = available_tool_observations(state["messages"], state)
        self.assertEqual([row["tool"] for row in observations],
                         ["get_order_info", "get_product_info", "search_product_manual"])
        self.assertEqual(observations[-1]["data"]["manual_id"], case["manual_id"])
        self.assertEqual(observations[-1]["data"]["source_tier"], "legacy_category")
        self.assertTrue(any(row["chunk_id"] == case["manual_chunk_id"]
                            for row in observations[-1]["data"]["results"]))
        self.assertEqual(state["messages"][-1].content, case["reference_response"])

    def test_order_product_manual_and_reference_response_share_one_snapshot(self):
        source_hash = hash_file(SOURCE)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.sqlite"
            manifest = build(path)
            self.assertEqual(hash_file(SOURCE), source_hash)
            with sqlite3.connect(path) as db:
                self.assertEqual(db.execute("PRAGMA foreign_key_check").fetchall(), [])
                self.assertEqual(db.execute("SELECT synthetic_fields_json FROM "
                    "agent_full_chain_fixture_metadata").fetchone()[0], '["orders.*"]')
            service = ManualSearchService(retrieval_strategy="bm25")
            context = RuntimeContext(
                connection_factory=lambda: configure_connection(sqlite3.connect(path)),
                default_snapshot_id=manifest["snapshot_id"],
                visible_snapshot_ids=(manifest["snapshot_id"],),
                manual_service=service,
            )
            llmb = ScriptedB(manifest["reference_response"])
            graph = build_agent(ScriptedA(manifest["order_id"], manifest["product_id"],
                                          manifest["sku_id"]), llmb)
            state = graph.invoke(make_initial_state(
                [HumanMessage(content=manifest["question"])], context=context), context=context)
            observations = available_tool_observations(state["messages"], state)
            self.assertEqual([row["tool"] for row in observations],
                             ["get_order_info", "get_product_info", "search_product_manual"])
            order, product, manual = observations
            self.assertEqual(order["data"]["order_id"], manifest["order_id"])
            self.assertEqual(order["data"]["product_id"], manifest["product_id"])
            self.assertEqual(product["data"]["sku_id"], manifest["sku_id"])
            self.assertEqual(manual["data"]["manual_id"], manifest["manual_id"])
            self.assertEqual({ref["snapshot_id"] for ref in order["evidence_refs"]},
                             {manifest["snapshot_id"]})
            self.assertEqual(manual["data"]["snapshot_id"], manifest["snapshot_id"])
            self.assertTrue(all(ref["manual_id"] == manifest["manual_id"]
                                for ref in manual["evidence_refs"]))
            self.assertEqual(state["messages"][-1].content, manifest["reference_response"])
            self.assertIn(manifest["order_id"], llmb.selected)
            self.assertIn(manifest["manual_id"], llmb.selected)
            self.assertEqual(len([m for m in state["messages"] if isinstance(m, ToolMessage)]), 4)


if __name__ == "__main__":
    unittest.main()
