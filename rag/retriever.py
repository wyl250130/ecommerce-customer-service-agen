#!/usr/bin/env python3
"""Retrieve chunks only within an already-bound product manual."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / "data/rag_index/manual_v1_tfidf_char"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class ManualRetriever:
    def __init__(self, index_dir: Path | str = DEFAULT_INDEX):
        self.index_dir = Path(index_dir)
        self.manifest = json.loads(
            (self.index_dir / "index_manifest.json").read_text(encoding="utf-8")
        )
        self.embeddings = np.load(
            self.index_dir / self.manifest["embeddings_file"], mmap_mode="r"
        )
        self.metadata = read_jsonl(self.index_dir / self.manifest["metadata_file"])
        if self.embeddings.ndim != 2 or len(self.metadata) != self.embeddings.shape[0]:
            raise ValueError("embedding and metadata row counts differ")
        for index, row in enumerate(self.metadata):
            if row.get("embedding_index") != index:
                raise ValueError(f"metadata row {index} has misaligned embedding_index")

        self.product_indices: dict[str, np.ndarray] = {}
        for index, row in enumerate(self.metadata):
            self.product_indices.setdefault(row["product_id"], []).append(index)
        self.product_indices = {
            product_id: np.asarray(indices, dtype=np.int64)
            for product_id, indices in self.product_indices.items()
        }
        self._load_encoder()

    def _load_encoder(self) -> None:
        config = self.manifest["backend_config"]
        backend = config["backend"]
        if backend == "tfidf_char":
            import joblib

            self.vectorizer = joblib.load(self.index_dir / config["artifact"])
            self.model = None
        elif backend == "sentence_transformer":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("sentence-transformers is required to query this index") from exc
            self.model = SentenceTransformer(config["model_name"])
            self.vectorizer = None
        else:
            raise ValueError(f"unsupported index backend: {backend}")

    def encode_query(self, query: str) -> np.ndarray:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        backend = self.manifest["backend_config"]["backend"]
        query = self.manifest["backend_config"].get("query_prefix", "") + query
        if backend == "tfidf_char":
            vector = self.vectorizer.transform([query]).toarray()[0].astype(np.float32)
            norm = float(np.linalg.norm(vector))
            return vector if norm == 0 else vector / norm
        vector = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0].astype(np.float32)
        return vector

    def search(self, product_id: str, query: str, top_k: int = 3) -> list[dict]:
        if not product_id:
            raise ValueError("product_id is required; this baseline does not infer products")
        if product_id not in self.product_indices:
            known = ", ".join(sorted(self.product_indices))
            raise KeyError(f"unknown product_id {product_id!r}; known products: {known}")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        candidate_indices = self.product_indices[product_id]
        query_vector = self.encode_query(query)
        candidate_matrix = np.asarray(self.embeddings[candidate_indices], dtype=np.float32)
        scores = candidate_matrix @ query_vector
        order = np.argsort(-scores, kind="stable")[: min(top_k, len(candidate_indices))]
        results = []
        for rank, local_index in enumerate(order, 1):
            global_index = int(candidate_indices[local_index])
            row = self.metadata[global_index]
            results.append({
                "rank": rank,
                "chunk_id": row["chunk_id"],
                "source_chunk_id": row["source_chunk_id"],
                "score": float(scores[local_index]),
                "manual_id": row["manual_id"],
                "product_id": row["product_id"],
                "section_title": row["section_title"],
                "content": row["content"],
            })
        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--product-id", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retriever = ManualRetriever(args.index_dir)
    print(json.dumps(
        retriever.search(args.product_id, args.query, args.top_k),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
