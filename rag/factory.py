"""Create a product-bound retrieval pipeline from application configuration."""

from __future__ import annotations

from collections.abc import Sequence

from .direct_cross_encoder import DirectCrossEncoderRetriever, HuggingFaceCrossEncoder
from .pipeline import RetrievalPipeline
from .retrievers.bm25 import BM25Retriever
from .retrievers.dense import DenseRetriever
from .retrievers.hybrid import HybridRetriever
from .rerankers.cross_encoder import CrossEncoderReranker


def create_retriever(
    strategy: str = "direct_cross_encoder", *, chunks: Sequence[dict] | None = None,
    scorer=None, index_dir=None, reranker: str | None = None,
    reranker_scorer=None, candidate_k: int | None = None,
    passage_template: str = "full", expected_candidates: int = 6,
) -> RetrievalPipeline:
    """Build once at application startup; search() always stays product-bound.

    Dense and hybrid need an existing index_dir. An injected scorer lets tests
    and applications control model loading; otherwise the approved HF model is
    loaded only when the chosen strategy needs it.
    """
    if strategy == "direct_cross_encoder":
        if chunks is None:
            raise ValueError("direct_cross_encoder requires chunks")
        model = scorer or HuggingFaceCrossEncoder()
        first_stage = DirectCrossEncoderRetriever(
            chunks, model.score_pairs, passage_template, expected_candidates)
    elif strategy == "bm25":
        if chunks is None:
            raise ValueError("bm25 requires chunks")
        first_stage = BM25Retriever(chunks)
    elif strategy == "dense":
        if index_dir is None:
            raise ValueError("dense requires index_dir")
        first_stage = DenseRetriever(index_dir)
    elif strategy == "hybrid":
        if chunks is None or index_dir is None:
            raise ValueError("hybrid requires chunks and index_dir")
        first_stage = HybridRetriever(BM25Retriever(chunks), DenseRetriever(index_dir))
    else:
        raise ValueError(f"unknown retrieval strategy: {strategy}")

    if reranker is None:
        second_stage = None
    elif reranker == "cross_encoder":
        model = reranker_scorer or scorer or HuggingFaceCrossEncoder()
        second_stage = CrossEncoderReranker(model.score_pairs)
    else:
        raise ValueError(f"unknown reranker: {reranker}")
    return RetrievalPipeline(first_stage, second_stage, candidate_k)
