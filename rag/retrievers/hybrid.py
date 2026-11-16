"""Fuse two product-bound ranked lists with reciprocal rank fusion."""

from __future__ import annotations

from ..base import Retriever
from ..schemas import validate_results


class HybridRetriever:
    def __init__(self, sparse: Retriever, dense: Retriever, fusion_k: int = 60):
        if fusion_k < 1:
            raise ValueError("fusion_k must be positive")
        self.sparse, self.dense, self.fusion_k = sparse, dense, fusion_k

    def search(self, product_id: str, query: str, top_k: int):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        rankings = [self.sparse.search(product_id, query, top_k),
                    self.dense.search(product_id, query, top_k)]
        combined = {}
        for ranking in rankings:
            for rank, row in enumerate(validate_results(ranking, product_id, top_k), 1):
                chunk_id = row["chunk_id"]
                if chunk_id not in combined:
                    combined[chunk_id] = [dict(row), 0.0]
                elif combined[chunk_id][0]["manual_id"] != row["manual_id"]:
                    raise ValueError("retrievers disagree about manual_id")
                combined[chunk_id][1] += 1 / (self.fusion_k + rank)
        ordered = sorted(combined.values(), key=lambda item: (-item[1], item[0]["chunk_id"]))
        return [{**row, "rank": rank, "score": score}
                for rank, (row, score) in enumerate(ordered[:top_k], 1)]
