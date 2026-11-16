"""Stage-A contract tests for the bound manual-search adapter."""

import json
import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path

from agent.recorder import TrajectoryRecorder
from agent.runner import AgentRunner
from agent.schemas import ToolAction
from agent.session_state import SessionState
from agent.tool_registry import execute_tool
from agent.tools.manual import DEFAULT_CHUNKS, DEFAULT_MANIFEST, DEFAULT_MANUALS, ManualSearchService
from agent.verifier import verify_trajectory


class FakeScorer:
    model_name = "fake-scorer"
    revision = "test"
    max_length = 512
    device = "cpu"

    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    def score_pairs(self, pairs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("inference unavailable")
        return [float(i) for i in range(len(pairs))]


class ManualToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with DEFAULT_MANUALS.open(encoding="utf-8") as handle:
            cls.manual = json.loads(next(handle))
        cls.binding = cls.manual["product_binding"]

    def setUp(self):
        self.scorer = FakeScorer()
        self.service = ManualSearchService(scorer=self.scorer)
        self.state = SessionState(
            conversation_id="test",
            snapshot_id=self.binding["snapshot_id"],
            current_product_id=self.binding["product_id"],
            current_sku_id=self.binding["sku_id"],
        )
        self.conn = sqlite3.connect(":memory:")

    def tearDown(self):
        self.conn.close()

    def call(self, arguments=None, state=None):
        return execute_tool(
            ToolAction("search_product_manual", arguments or {"query": "怎么使用？"}),
            state or self.state, self.conn, self.service,
        )

    def test_found_is_bound_and_auditable(self):
        result = self.call()
        self.assertTrue(result.ok)
        self.assertEqual(result.status, "found")
        self.assertEqual(result.data["manual_id"], self.manual["manual_id"])
        self.assertEqual(result.data["snapshot_id"], self.binding["snapshot_id"])
        self.assertEqual(len(result.data["results"]), 5)
        self.assertEqual([row["score"] for row in result.data["results"]], [5., 4., 3., 2., 1.])
        self.assertEqual(len(result.evidence_refs), 5)
        self.assertEqual(len({row["section_id"] for row in result.data["results"]}), 5)
        self.assertTrue(all(ref["manual_id"] == self.manual["manual_id"] for ref in result.evidence_refs))
        self.assertTrue(all(row["content"] and row["chunk_sha256"] for row in result.data["results"]))
        self.assertEqual(self.scorer.calls, 1)

    def test_missing_product_and_sku_require_clarification(self):
        self.state.current_product_id = None
        result = self.call()
        self.assertEqual((result.status, result.error_code), ("partial", "PRODUCT_NOT_BOUND"))
        self.state.current_product_id = self.binding["product_id"]
        self.state.current_sku_id = None
        result = self.call()
        self.assertEqual((result.status, result.error_code), ("partial", "SKU_NOT_BOUND"))
        self.assertEqual(self.scorer.calls, 0)

    def test_wrong_snapshot_or_product_never_retrieves(self):
        self.state.snapshot_id = "OTHER-SNAPSHOT"
        self.assertEqual(self.call().status, "no_match")
        self.state.snapshot_id = self.binding["snapshot_id"]
        self.state.current_product_id = "OTHER-PRODUCT"
        self.assertEqual(self.call().status, "no_match")
        self.assertEqual(self.scorer.calls, 0)

    def test_model_cannot_choose_scope_or_configuration(self):
        for key in ("product_id", "sku_id", "snapshot_id", "manual_id", "top_k"):
            with self.subTest(key=key):
                self.assertEqual(self.call({"query": "怎么用？", key: "override"}).status, "forbidden")
        self.assertEqual(self.scorer.calls, 0)

    def test_query_is_the_only_required_argument(self):
        self.assertEqual(self.call({"query": ""}).error_code, "QUERY_REQUIRED")
        self.assertEqual(self.call({"query": "x", "extra": 1}).error_code, "ARGUMENT_ERROR")
        self.assertEqual(self.scorer.calls, 0)

    def test_scorer_failure_is_tool_error(self):
        self.service = ManualSearchService(scorer=FakeScorer(fail=True))
        result = self.call()
        self.assertEqual((result.status, result.error_code), ("tool_error", "MANUAL_SEARCH_FAILED"))
        self.assertFalse(result.ok)

    def test_top_k_variants_and_shared_service(self):
        for k in (3, 5, 6):
            service = ManualSearchService(scorer=self.scorer, top_k=k)
            result = service.search(
                snapshot_id=self.state.snapshot_id,
                product_id=self.state.current_product_id,
                sku_id=self.state.current_sku_id,
                query="两个章节的信息",
            )
            self.assertEqual(len(result.data["results"]), k)
            self.assertEqual(result.data["retriever_config"]["top_k"], k)
        self.assertEqual(self.scorer.calls, 3)

    def test_bm25_strategy_preserves_frozen_manual_binding(self):
        service = ManualSearchService(retrieval_strategy="bm25")
        result = service.search(
            snapshot_id=self.state.snapshot_id,
            product_id=self.state.current_product_id,
            sku_id=self.state.current_sku_id,
            query="怎么使用？",
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.data["retriever_config"]["retriever_backend"], "bm25")
        self.assertEqual(len(result.data["results"]), 5)
        self.assertTrue(all(ref["manual_id"] == self.manual["manual_id"]
                            for ref in result.evidence_refs))

    def test_dense_index_must_match_frozen_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "index_manifest.json").write_text(
                json.dumps({"chunks_sha256": "wrong"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "index does not match"):
                ManualSearchService(retrieval_strategy="dense", index_dir=Path(directory))

    def test_runner_records_manual_evidence_without_claiming_answer_sufficiency(self):
        runner = AgentRunner(self.conn, manual_service=self.service)
        result = runner.run(self.state, [
            {"tool": "search_product_manual", "arguments": {"query": "怎么使用？"}},
            {"type": "ask_user", "question": "还需要确认什么？"},
        ])
        self.assertEqual(result.status, "needs_user")
        self.assertEqual(self.state.evidence[0].source_type, "product_manual")
        self.assertEqual(self.state.evidence[0].source_id, self.manual["manual_id"])
        self.assertEqual(len(self.state.tool_history), 1)

    def test_partial_can_continue_to_ask_user_and_verify(self):
        self.state.current_product_id = None
        recorder = TrajectoryRecorder("manual-partial")
        result = AgentRunner(self.conn, manual_service=self.service).run(self.state, [
            {"tool": "search_product_manual", "arguments": {"query": "怎么用？"}},
            {"type": "ask_user", "question": "请问是哪件商品？"},
        ], recorder=recorder)
        self.assertEqual(result.status, "needs_user")
        self.assertEqual(self.state.tool_history[0]["result"]["status"], "partial")
        verification = verify_trajectory(recorder.to_dict(), self.state, "manual_usage")
        self.assertTrue(verification.passed, verification.errors)

    def test_no_match_can_continue_to_handoff(self):
        self.state.snapshot_id = "OTHER-SNAPSHOT"
        result = AgentRunner(self.conn, manual_service=self.service).run(self.state, [
            {"tool": "search_product_manual", "arguments": {"query": "怎么用？"}},
            {"type": "handoff"},
        ])
        self.assertEqual(result.status, "handoff")
        self.assertEqual(self.state.tool_history[0]["result"]["status"], "no_match")

    def test_verifier_rejects_model_supplied_scope(self):
        for key in ("product_id", "sku_id", "snapshot_id", "manual_id", "top_k"):
            with self.subTest(key=key):
                trajectory = {"steps": [{"action": {
                    "type": "tool_call", "tool": "search_product_manual",
                    "arguments": {"query": "怎么用？", key: "override"},
                }}]}
                result = verify_trajectory(trajectory, self.state, "manual_usage")
                self.assertFalse(result.passed)
                self.assertIn(key, result.errors[0])

    def test_changed_manifest_or_chunk_cannot_start(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
            manifest["corpus_sha256"] = "0" * 64
            manifest_path = temp / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest"):
                ManualSearchService(manifest_path=manifest_path, scorer=self.scorer)

            manifest_path.write_text(DEFAULT_MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")
            chunk_path = temp / DEFAULT_CHUNKS.name
            chunk_path.write_text(DEFAULT_CHUNKS.read_text(encoding="utf-8").replace("使用前检查", "使用前检查已修改", 1), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "chunks file SHA-256 mismatch"):
                ManualSearchService(chunks_path=chunk_path, manifest_path=manifest_path, scorer=self.scorer)

    def test_changed_chunk_with_recomputed_self_hash_cannot_start(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            manifest_path = temp / "manifest.json"
            manifest_path.write_text(DEFAULT_MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")
            rows = [json.loads(line) for line in DEFAULT_CHUNKS.read_text(encoding="utf-8").splitlines()]
            rows[0]["content"] += "（被修改）"
            canonical = json.dumps(
                {key: value for key, value in rows[0].items() if key != "chunk_sha256"},
                ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")
            rows[0]["chunk_sha256"] = hashlib.sha256(canonical).hexdigest()
            chunk_path = temp / DEFAULT_CHUNKS.name
            chunk_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "chunks file SHA-256 mismatch"):
                ManualSearchService(chunks_path=chunk_path, manifest_path=manifest_path, scorer=self.scorer)


if __name__ == "__main__":
    unittest.main()
