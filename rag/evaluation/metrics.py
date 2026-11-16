"""Simple per-query retrieval metrics; empty-gold rows are excluded."""

from __future__ import annotations

from collections.abc import Iterable


def evaluate_ranking(retrieved_ids: Iterable[str], gold_ids: Iterable[str], k: int) -> dict[str, float]:
    if k < 1:
        raise ValueError("k must be positive")
    gold = set(gold_ids)
    if not gold:
        raise ValueError("gold_ids must be non-empty; evaluate no-answer separately")
    hits = list(retrieved_ids)[:k]
    first = next((rank for rank, chunk_id in enumerate(hits, 1) if chunk_id in gold), None)
    return {
        "recall_at_k": float(first is not None),
        "evidence_recall_at_k": len(gold.intersection(hits)) / len(gold),
        "mrr_at_k": 1 / first if first else 0.0,
    }
