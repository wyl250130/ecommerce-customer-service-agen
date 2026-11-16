#!/usr/bin/env python3
"""Build a small NumPy index for the frozen product-manual chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = ROOT / "data/processed/product_manuals/frozen_v1/product_manual_chunks_frozen_v1.jsonl"
DEFAULT_INDEX = ROOT / "data/rag_index/manual_v1_tfidf_char"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def embedding_text(chunk: dict) -> str:
    return f"商品：{chunk['product_name']}\n章节：{chunk['section_title']}\n内容：{chunk['content']}"


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def encode_tfidf_char(texts: list[str], output_dir: Path) -> tuple[np.ndarray, dict]:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 3),
        min_df=1,
        max_df=1.0,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(texts).toarray().astype(np.float32)
    matrix = l2_normalize(matrix).astype(np.float32)
    artifact = output_dir / "tfidf_vectorizer.joblib"
    joblib.dump(vectorizer, artifact)
    return matrix, {
        "backend": "tfidf_char",
        "artifact": artifact.name,
        "analyzer": "char",
        "ngram_range": [1, 3],
        "note": "Offline lexical vector baseline; not a pretrained semantic embedding model.",
    }


def encode_sentence_transformer(
    texts: list[str], output_dir: Path, model_name: str, batch_size: int
) -> tuple[np.ndarray, dict]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. Install it in the target environment "
            "or use --backend tfidf_char."
        ) from exc

    model = SentenceTransformer(model_name)
    matrix = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    return matrix, {
        "backend": "sentence_transformer",
        "model_name": model_name,
        "batch_size": batch_size,
        "note": "Dense pretrained embedding backend.",
    }


def build_index(
    chunks_path: Path,
    output_dir: Path,
    backend: str = "tfidf_char",
    model_name: str = "BAAI/bge-small-zh-v1.5",
    batch_size: int = 32,
    query_prefix: str = "",
) -> dict:
    chunks = read_jsonl(chunks_path)
    if not chunks:
        raise ValueError("chunk file is empty")
    required = {
        "manual_id", "product_id", "product_name", "chunk_id", "section_id",
        "section_title", "content",
    }
    for index, chunk in enumerate(chunks):
        missing = required - set(chunk)
        if missing:
            raise ValueError(f"chunk {index} missing fields: {sorted(missing)}")
    if len({row["chunk_id"] for row in chunks}) != len(chunks):
        raise ValueError("chunk_id must be globally unique")

    output_dir.mkdir(parents=True, exist_ok=True)
    texts = [embedding_text(chunk) for chunk in chunks]
    if backend == "tfidf_char":
        embeddings, backend_config = encode_tfidf_char(texts, output_dir)
    elif backend == "sentence_transformer":
        embeddings, backend_config = encode_sentence_transformer(
            texts, output_dir, model_name, batch_size
        )
    else:
        raise ValueError(f"unsupported backend: {backend}")
    backend_config["query_prefix"] = query_prefix
    backend_config["passage_prefix"] = ""

    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise ValueError("embedding matrix shape does not match chunk count")

    embeddings_path = output_dir / "manual_embeddings.npy"
    np.save(embeddings_path, embeddings)
    metadata = []
    for embedding_index, chunk in enumerate(chunks):
        metadata.append({
            "embedding_index": embedding_index,
            "manual_id": chunk["manual_id"],
            "manual_version": chunk.get("manual_version"),
            "product_id": chunk["product_id"],
            "product_name": chunk["product_name"],
            "sku_id": chunk.get("sku_id"),
            "source_chunk_id": chunk["chunk_id"],
            "chunk_id": chunk["section_id"],
            "section_title": chunk["section_title"],
            "content": chunk["content"],
        })
    metadata_path = output_dir / "manual_metadata.jsonl"
    write_jsonl(metadata_path, metadata)

    product_counts: dict[str, int] = {}
    for chunk in chunks:
        product_counts[chunk["product_id"]] = product_counts.get(chunk["product_id"], 0) + 1
    manifest = {
        "index_version": "manual_rag_baseline_v1",
        "chunks_path": portable_path(chunks_path),
        "chunks_sha256": file_sha256(chunks_path),
        "chunk_count": len(chunks),
        "embedding_dim": int(embeddings.shape[1]),
        "embeddings_file": embeddings_path.name,
        "metadata_file": metadata_path.name,
        "normalized": True,
        "embedding_text_template": "商品：{product_name}\\n章节：{section_title}\\n内容：{content}",
        "product_chunk_counts": dict(sorted(product_counts.items())),
        "backend_config": backend_config,
    }
    (output_dir / "index_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_INDEX)
    parser.add_argument(
        "--backend", choices=("tfidf_char", "sentence_transformer"), default="tfidf_char"
    )
    parser.add_argument("--model-name", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--query-prefix",
        default="",
        help="Prefix applied only when encoding queries; passage embeddings remain unchanged.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_index(
        args.chunks, args.output_dir, args.backend, args.model_name, args.batch_size,
        args.query_prefix,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
