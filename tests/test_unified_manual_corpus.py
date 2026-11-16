import json
import tempfile
import unittest
from pathlib import Path

from rag.unified_manuals import UnifiedManualCorpus
from scripts.build_manual_corpus_v2 import build


class UnifiedManualCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temporary.name)
        cls.manifest = build(output_dir=cls.directory)
        cls.corpus = UnifiedManualCorpus(cls.directory)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_source_counts_and_reproducible_manifest(self):
        self.assertEqual(self.manifest["source_counts"], {
            "legacy_manuals": 8, "frozen_manuals": 8,
            "legacy_bindings": 236, "frozen_bindings": 8,
            "verified_derived_bindings": 1,
        })
        before = self.manifest["output_sha256"]
        self.assertEqual(build(output_dir=self.directory)["output_sha256"], before)
        self.assertEqual(self.manifest["counts"]["human_queries"], 800)

    def test_exact_binding_and_preferred_frozen_manual(self):
        snapshot = "GOLD-SNAP-SEMANTIC-GOLD-TRAIN-000551"
        product, sku = "BABY-COTTON-SWAB-551", "FINE-TIP"
        preferred = self.corpus.candidates(snapshot, product, sku)
        self.assertEqual(len(preferred), 6)
        self.assertEqual({row["source_tier"] for row in preferred}, {"frozen_product"})
        overlap = self.corpus.candidates(snapshot, product, sku, include_legacy_overlap=True)
        self.assertGreater(len(overlap), len(preferred))
        self.assertEqual(self.corpus.candidates(snapshot, "WRONG", sku), [])
        self.assertEqual(self.corpus.candidates("WRONG", product, sku), [])
        self.assertEqual(self.corpus.candidates(snapshot, product, "WRONG"), [])

    def test_chat_fixture_inherits_only_verified_category_binding(self):
        rows = self.corpus.search("CHAT-E2E-SNAPSHOT-001",
                                  "context-product-GOLD-DEV-000063", "snack-durian",
                                  "冻干榴莲开封后怎么保存？", top_k=3)
        self.assertTrue(rows)
        self.assertEqual({row["manual_id"] for row in rows}, {"MANUAL-DRIED-FRUIT-001"})
        self.assertEqual({row["binding_source"] for row in rows},
                         {"verified_chat_fixture_inheritance"})
        self.assertEqual(self.corpus.candidates("CHAT-E2E-SNAPSHOT-001",
                          "context-product-GOLD-DEV-000063", "OTHER"), [])

    def test_human_gold_references_bound_chunks(self):
        with (self.directory / "human_queries.jsonl").open(encoding="utf-8") as source:
            queries = [json.loads(line) for line in source]
        for row in queries:
            snapshot, product, sku = json.loads(row["binding_key"])
            candidates = self.corpus.candidates(snapshot, product, sku)
            self.assertEqual({item["manual_id"] for item in candidates}, {row["manual_id"]})
            self.assertTrue(set(row["gold_chunk_ids"]).issubset(
                {item["chunk_id"] for item in candidates}))


if __name__ == "__main__":
    unittest.main()
