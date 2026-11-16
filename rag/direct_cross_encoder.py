#!/usr/bin/env python3
"""Direct product-bound scoring with BAAI/bge-reranker-v2-m3."""
from __future__ import annotations

from typing import Callable, Sequence

FULL_TEMPLATE = "商品：{product_name}\n章节：{section_title}\n内容：{content}"
MODEL_NAME = "BAAI/bge-reranker-v2-m3"
MODEL_REVISION = "995ec6ee29e8a96b27eee66c584c4340104ab8e5"


def passage_text(chunk: dict, template: str = "full") -> str:
    if template == "full":
        return FULL_TEMPLATE.format(
            product_name=chunk["product_name"],
            section_title=chunk["section_title"],
            content=chunk["content"],
        )
    if template == "content_only":
        return chunk["content"]
    raise ValueError(f"unsupported passage template: {template}")


def normalize_chunk(chunk: dict) -> dict:
    """Map frozen chunk schema (section_id) to the evaluator's chunk_id."""
    row = dict(chunk)
    row["source_chunk_id"] = chunk.get("chunk_id")
    row["chunk_id"] = chunk.get("section_id") or chunk.get("chunk_id")
    if not row["chunk_id"]:
        raise ValueError("chunk has neither chunk_id nor section_id")
    return row


class DirectCrossEncoderRetriever:
    """Score every chunk for a product; scorer accepts a list of text pairs."""

    def __init__(
        self,
        chunks: Sequence[dict],
        score_pairs: Callable[[list[list[str]]], Sequence[float]],
        passage_template: str = "full",
        expected_candidates: int = 6,
    ):
        self.chunks = [normalize_chunk(chunk) for chunk in chunks]
        self.score_pairs = score_pairs
        self.passage_template = passage_template
        self.expected_candidates = expected_candidates
        self.by_product: dict[str, list[dict]] = {}
        for chunk in self.chunks:
            self.by_product.setdefault(chunk["product_id"], []).append(chunk)
        for product_id, candidates in self.by_product.items():
            ids = [chunk["chunk_id"] for chunk in candidates]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate chunk_id for product {product_id}")

    def candidate_pairs(self, product_id: str, query: str) -> tuple[list[dict], list[list[str]]]:
        if not product_id:
            raise ValueError("product_id is required")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if product_id not in self.by_product:
            raise KeyError(f"unknown product_id: {product_id}")
        candidates = self.by_product[product_id]
        if len(candidates) != self.expected_candidates:
            raise ValueError(
                f"product {product_id} has {len(candidates)} candidates; "
                f"expected {self.expected_candidates}"
            )
        if any(row["product_id"] != product_id for row in candidates):
            raise AssertionError("candidate product filter leaked")
        pairs = [[query, passage_text(row, self.passage_template)] for row in candidates]
        return candidates, pairs

    def search(self, product_id: str, query: str, top_k: int = 6) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        candidates, pairs = self.candidate_pairs(product_id, query)
        scores = list(self.score_pairs(pairs))
        if len(scores) != len(candidates):
            raise ValueError("scorer returned a different number of scores than input pairs")
        order = sorted(range(len(scores)), key=lambda i: -float(scores[i]))
        results = []
        for rank, index in enumerate(order[: min(top_k, len(order))], 1):
            row = candidates[index]
            results.append({
                "rank": rank,
                "chunk_id": row["chunk_id"],
                "source_chunk_id": row.get("source_chunk_id") or row.get("chunk_id"),
                "score": float(scores[index]),
                "manual_id": row["manual_id"],
                "product_id": row["product_id"],
                "section_title": row["section_title"],
                "content": row["content"],
            })
        return results


class HuggingFaceCrossEncoder:
    """Transformers implementation; truncates passage only and keeps query intact."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        revision: str = MODEL_REVISION,
        max_length: int = 512,
        batch_size: int = 6,
        device: str = "auto",
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.model_name = model_name
        self.revision = revision
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = self._select_device(device)
        load_args = {"revision": revision, "cache_dir": cache_dir, "local_files_only": local_files_only}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **load_args)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, **load_args)
        self.model.to(self.device)
        self.model.eval()

    def _select_device(self, requested: str) -> str:
        if requested != "auto":
            if requested == "mps" and not self.torch.backends.mps.is_available():
                raise RuntimeError("MPS was requested but is unavailable in this PyTorch build")
            if requested.startswith("cuda") and not self.torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable")
            return requested
        if self.torch.cuda.is_available():
            return "cuda"
        if getattr(self.torch.backends, "mps", None) and self.torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def token_length(self, query: str, passage: str) -> int:
        encoded = self.tokenizer(query, passage, add_special_tokens=True, truncation=False)
        return len(encoded["input_ids"])

    def score_pairs(self, pairs: list[list[str]]) -> list[float]:
        all_scores: list[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            inputs = self.tokenizer(
                [pair[0] for pair in batch],
                [pair[1] for pair in batch],
                padding=True,
                truncation="only_second",
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self.torch.inference_mode():
                logits = self.model(**inputs, return_dict=True).logits.reshape(-1).float()
            all_scores.extend(logits.detach().cpu().tolist())
        return all_scores

    def synchronize(self) -> None:
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)
        elif self.device == "mps":
            self.torch.mps.synchronize()
