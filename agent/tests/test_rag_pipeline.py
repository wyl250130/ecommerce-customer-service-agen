import unittest

from rag.factory import create_retriever
from rag.retrievers.hybrid import HybridRetriever
from rag.evaluation.metrics import evaluate_ranking


CHUNKS = [
    {"section_id": "A", "chunk_id": "source-A", "manual_id": "M",
     "product_id": "P", "product_name": "样品", "section_title": "清洁",
     "content": "清洁前断开电源"},
    {"section_id": "B", "chunk_id": "source-B", "manual_id": "M",
     "product_id": "P", "product_name": "样品", "section_title": "安装",
     "content": "安装时固定底座"},
]


class Scorer:
    def score_pairs(self, pairs):
        return [float(i) for i in range(len(pairs))]


class PipelineTests(unittest.TestCase):
    def test_default_direct_uses_existing_retriever(self):
        pipeline = create_retriever(chunks=CHUNKS, scorer=Scorer(), expected_candidates=2)
        self.assertEqual([row["chunk_id"] for row in pipeline.search("P", "如何安装", 2)], ["B", "A"])

    def test_bm25_and_optional_reranker(self):
        plain = create_retriever("bm25", chunks=CHUNKS)
        self.assertEqual(plain.search("P", "清洁", 1)[0]["chunk_id"], "A")
        ranked = create_retriever("bm25", chunks=CHUNKS, reranker="cross_encoder",
                                  reranker_scorer=Scorer(), candidate_k=2)
        self.assertEqual(ranked.search("P", "清洁", 1)[0]["chunk_id"], "B")

    def test_hybrid_and_product_scope(self):
        sparse = create_retriever("bm25", chunks=CHUNKS)
        hybrid = HybridRetriever(sparse, sparse)
        self.assertEqual(hybrid.search("P", "安装", 1)[0]["product_id"], "P")
        with self.assertRaises(KeyError):
            hybrid.search("OTHER", "安装", 1)

    def test_metrics(self):
        self.assertEqual(evaluate_ranking(["A", "B"], ["B"], 2)["mrr_at_k"], 0.5)


if __name__ == "__main__":
    unittest.main()
