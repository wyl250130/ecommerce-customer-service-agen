"""Small contracts shared by all product-bound retrieval strategies."""

from __future__ import annotations

from typing import Protocol, Sequence

from .schemas import RetrievalResult


class Retriever(Protocol):
    def search(self, product_id: str, query: str, top_k: int) -> list[RetrievalResult]: ...


class Reranker(Protocol):
    def rerank(self, query: str, candidates: Sequence[RetrievalResult], top_k: int) -> list[RetrievalResult]: ...
