"""Small in-memory BM25 over the already-approved product chunks."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from ..direct_cross_encoder import normalize_chunk
from ..schemas import RetrievalResult


def _tokens(text: str) -> list[str]:
    # Character bigrams handle Chinese text without an additional tokenizer.
    text = re.sub(r"\s+", "", text.lower())
    return [text[i:i + 2] for i in range(max(0, len(text) - 1))] or ([text] if text else [])


class BM25Retriever:
    def __init__(self, chunks: Sequence[dict], k1: float = 1.2, b: float = 0.75):
        self.by_product: dict[str, list[dict]] = {}
        for chunk in chunks:
            row = normalize_chunk(chunk)
            self.by_product.setdefault(row["product_id"], []).append(row)
        self.k1, self.b = k1, b

    def search(self, product_id: str, query: str, top_k: int) -> list[RetrievalResult]:
        if top_k < 1 or not query.strip():
            raise ValueError("positive top_k and non-empty query are required")
        rows = self.by_product.get(product_id)
        if rows is None:
            raise KeyError(f"unknown product_id: {product_id}")
        terms = set(_tokens(query))
        docs = [Counter(_tokens(row["section_title"] + row["content"])) for row in rows]
        lengths = [sum(doc.values()) for doc in docs]
        average = sum(lengths) / len(lengths) if lengths else 1
        frequencies = {term: sum(term in doc for doc in docs) for term in terms}
        scored = []
        for index, doc in enumerate(docs):
            score = 0.0
            for term in terms:
                tf = doc[term]
                if tf:
                    df = frequencies[term]
                    idf = math.log(1 + (len(docs) - df + 0.5) / (df + 0.5))
                    score += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * lengths[index] / average))
            scored.append((index, score))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return [{"rank": rank, "chunk_id": row["chunk_id"],
                 "source_chunk_id": row.get("source_chunk_id") or row["chunk_id"],
                 "score": float(score), "manual_id": row["manual_id"],
                 "product_id": row["product_id"], "section_title": row["section_title"],
                 "content": row["content"]}
                for rank, (index, score) in enumerate(scored[:top_k], 1)
                for row in [rows[index]]]
