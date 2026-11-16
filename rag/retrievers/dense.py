"""Adapter for the existing indexed dense/TF-IDF ManualRetriever."""

from __future__ import annotations

class DenseRetriever:
    def __init__(self, index_dir):
        from ..retriever import ManualRetriever

        self.backend = ManualRetriever(index_dir)

    def search(self, product_id: str, query: str, top_k: int):
        return self.backend.search(product_id, query, top_k)
