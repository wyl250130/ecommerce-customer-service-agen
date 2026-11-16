"""The result shape exposed to manual search and evaluation."""

from __future__ import annotations

from typing_extensions import NotRequired, TypedDict


class RetrievalResult(TypedDict):
    rank: int
    chunk_id: str
    source_chunk_id: str
    score: float
    manual_id: str
    product_id: str
    section_title: str
    content: str
    retrieval_score: NotRequired[float]


def validate_results(results: list[dict], product_id: str, top_k: int) -> list[RetrievalResult]:
    """Check a strategy's boundary contract without changing its ranking."""
    if len(results) > top_k:
        raise ValueError("retriever returned more than top_k results")
    seen = set()
    for rank, row in enumerate(results, 1):
        if row.get("product_id") != product_id:
            raise ValueError("retriever returned a different product")
        chunk_id = row.get("chunk_id")
        if not chunk_id or chunk_id in seen:
            raise ValueError("retriever returned empty or duplicate chunk_id")
        seen.add(chunk_id)
        if row.get("rank") != rank:
            raise ValueError("retriever returned nonconsecutive ranks")
        if not all(key in row for key in ("manual_id", "source_chunk_id", "score", "section_title", "content")):
            raise ValueError("retriever result is missing required fields")
    return results
