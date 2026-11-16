"""Apply an injected cross-encoder scorer to retrieved candidates."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ..direct_cross_encoder import passage_text
from ..schemas import RetrievalResult


class CrossEncoderReranker:
    def __init__(self, score_pairs: Callable[[list[list[str]]], Sequence[float]],
                 passage_template: str = "content_only"):
        self.score_pairs = score_pairs
        self.passage_template = passage_template

    def rerank(self, query: str, candidates: Sequence[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        pairs = [[query, passage_text(row, self.passage_template)] for row in candidates]
        scores = list(self.score_pairs(pairs)) if pairs else []
        if len(scores) != len(candidates):
            raise ValueError("scorer returned a different number of scores than candidates")
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
        return [{**candidates[index], "rank": rank,
                 "retrieval_score": candidates[index]["score"], "score": float(scores[index])}
                for rank, index in enumerate(order[:top_k], 1)]
