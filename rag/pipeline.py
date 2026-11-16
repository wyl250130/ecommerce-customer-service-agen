"""Product-bound retrieval followed by an optional reranking stage."""

from __future__ import annotations

from .base import Reranker, Retriever
from .schemas import RetrievalResult, validate_results


class RetrievalPipeline:
    def __init__(self, retriever: Retriever, reranker: Reranker | None = None,
                 candidate_k: int | None = None):
        if candidate_k is not None and candidate_k < 1:
            raise ValueError("candidate_k must be positive")
        self.retriever = retriever
        self.reranker = reranker
        self.candidate_k = candidate_k

    def search(self, product_id: str, query: str, top_k: int) -> list[RetrievalResult]:
        if not product_id or not isinstance(query, str) or not query.strip() or top_k < 1:
            raise ValueError("product_id, non-empty query and positive top_k are required")
        fetch_k = max(top_k, self.candidate_k or top_k) if self.reranker else top_k
        candidates = validate_results(self.retriever.search(product_id, query, fetch_k), product_id, fetch_k)
        if self.reranker:
            return validate_results(self.reranker.rerank(query, candidates, top_k), product_id, top_k)
        return candidates
