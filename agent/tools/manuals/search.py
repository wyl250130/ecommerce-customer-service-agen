"""Bound, auditable adapter around the frozen manual Direct Cross-Encoder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rag.direct_cross_encoder import MODEL_NAME, MODEL_REVISION, HuggingFaceCrossEncoder, normalize_chunk
from rag.factory import create_retriever

from ...schemas import ToolResult


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHUNKS = ROOT / "data/processed/product_manuals/frozen_v1/product_manual_chunks_frozen_v1.jsonl"
DEFAULT_MANUALS = ROOT / "data/processed/product_manuals/frozen_v1/product_manuals_frozen_v1.jsonl"
DEFAULT_MANIFEST = ROOT / "data/processed/product_manuals/frozen_v1/manifest.json"
TOOL_NAME = "search_product_manual"
ALLOWED_TOP_K = {3, 5, 6}
# Pin the approved v1 corpus independently of its editable manifest.
FROZEN_V1_CORPUS_SHA256 = "c9f5b6ebfb13f753a7d548f350ffbd007221c3f055fa2e1ff7a605c7795d2fe7"
FROZEN_V1_CHUNKS_FILE_SHA256 = "3d2c5585088d355cd75120a03888a20709fce4f087ac03b6a6cf52891756500a"
FROZEN_V1_MANUALS_FILE_SHA256 = "b8a469b78e552262e6ff194c2348310434d9a18256fd5be7f76f25686633ffcf"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _failure(status: str, code: str, message: str, missing: list[str] | None = None) -> ToolResult:
    return ToolResult(
        ok=False, tool=TOOL_NAME, status=status, error_code=code,
        error_message=message, missing_fields=missing or [],
    )


class ManualSearchService:
    """One loaded model and corpus, shared across calls in an application instance.

    The v1 allowlist binds a manual only to its exact frozen source snapshot.
    Extending a manual to other snapshots requires an explicit reviewed binding.
    """

    def __init__(
        self,
        *,
        chunks_path: Path = DEFAULT_CHUNKS,
        manuals_path: Path = DEFAULT_MANUALS,
        manifest_path: Path = DEFAULT_MANIFEST,
        top_k: int = 5,
        max_length: int = 512,
        device: str = "auto",
        model_name: str = MODEL_NAME,
        model_revision: str = MODEL_REVISION,
        passage_template: str = "full",
        scorer: Any | None = None,
        retrieval_strategy: str = "direct_cross_encoder",
        index_dir: Path | None = None,
        reranker: str | None = None,
        candidate_k: int | None = None,
    ) -> None:
        if top_k not in ALLOWED_TOP_K:
            raise ValueError(f"top_k must be one of {sorted(ALLOWED_TOP_K)}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "frozen" or manifest.get("corpus_sha256") != FROZEN_V1_CORPUS_SHA256:
            raise ValueError("unapproved or changed frozen manual manifest")
        if manifest.get("manuals_file") != manuals_path.name or manifest.get("chunks_file") != chunks_path.name:
            raise ValueError("manual files do not match frozen manifest")
        self.top_k = top_k
        self.chunks_sha256 = _hash_file(chunks_path)
        self.manuals_sha256 = _hash_file(manuals_path)
        if self.chunks_sha256 != FROZEN_V1_CHUNKS_FILE_SHA256:
            raise ValueError("frozen manual chunks file SHA-256 mismatch")
        if self.manuals_sha256 != FROZEN_V1_MANUALS_FILE_SHA256:
            raise ValueError("frozen manuals file SHA-256 mismatch")
        if retrieval_strategy in {"dense", "hybrid"} and index_dir is not None:
            index_manifest = json.loads((Path(index_dir) / "index_manifest.json").read_text(encoding="utf-8"))
            if index_manifest.get("chunks_sha256") != self.chunks_sha256:
                raise ValueError("retrieval index does not match frozen manual chunks")
        manuals = _read_jsonl(manuals_path)
        chunks = _read_jsonl(chunks_path)
        if len(manuals) != manifest.get("manual_count") or len(chunks) != manifest.get("section_chunk_count"):
            raise ValueError("frozen manual/chunk count mismatch")
        if any(row.get("content_sha256") != _canonical_hash({k: v for k, v in row.items() if k != "content_sha256"}) for row in manuals):
            raise ValueError("frozen manual content hash mismatch")
        if _canonical_hash([row["content_sha256"] for row in manuals]) != FROZEN_V1_CORPUS_SHA256:
            raise ValueError("frozen manual corpus hash mismatch")
        if any(row.get("chunk_sha256") != _canonical_hash({k: v for k, v in row.items() if k != "chunk_sha256"}) for row in chunks):
            raise ValueError("frozen manual chunk hash mismatch")
        manual_hashes = {row["manual_id"]: row["content_sha256"] for row in manuals}
        if any(row.get("manual_content_sha256") != manual_hashes.get(row["manual_id"]) for row in chunks):
            raise ValueError("frozen chunk/manual hash mismatch")
        self.bindings: dict[tuple[str, str, str], str] = {}
        for manual in manuals:
            binding = manual["product_binding"]
            key = (binding["snapshot_id"], binding["product_id"], binding["sku_id"])
            if key in self.bindings:
                raise ValueError(f"duplicate manual binding: {key}")
            self.bindings[key] = manual["manual_id"]
        needs_scorer = retrieval_strategy == "direct_cross_encoder" or reranker == "cross_encoder"
        self.scorer = scorer or (HuggingFaceCrossEncoder(
            model_name=model_name, revision=model_revision, max_length=max_length,
            device=device,
        ) if needs_scorer else None)
        self.retriever = create_retriever(
            strategy=retrieval_strategy, chunks=chunks, scorer=self.scorer,
            index_dir=index_dir, reranker=reranker, candidate_k=candidate_k,
            passage_template=passage_template, expected_candidates=6,
        )
        self.chunks_by_product: dict[str, dict[str, dict]] = {}
        for chunk in chunks:
            row = normalize_chunk(chunk)
            self.chunks_by_product.setdefault(row["product_id"], {})[row["chunk_id"]] = row
        self.config = {
            "retriever_backend": retrieval_strategy,
            "model_name": getattr(self.scorer, "model_name", model_name if self.scorer else None),
            "model_revision": getattr(self.scorer, "revision", model_revision if self.scorer else None),
            "top_k": top_k,
            "max_length": getattr(self.scorer, "max_length", max_length if self.scorer else None),
            "device": getattr(self.scorer, "device", device if self.scorer else None),
            "passage_template": passage_template,
            "reranker": reranker,
            "candidate_k": candidate_k,
            "chunks_sha256": self.chunks_sha256,
            "manuals_sha256": self.manuals_sha256,
            "corpus_sha256": FROZEN_V1_CORPUS_SHA256,
        }
        for (snapshot_id, product_id, sku_id), manual_id in self.bindings.items():
            candidates = list(self.chunks_by_product.get(product_id, {}).values())
            if len(candidates) != 6 or any(
                row["manual_id"] != manual_id or row.get("sku_id") != sku_id
                for row in candidates
            ):
                raise ValueError(f"frozen binding/chunk mismatch: {snapshot_id}/{product_id}/{sku_id}")

    def search(
        self, *, snapshot_id: str, product_id: str | None,
        sku_id: str | None, query: str,
    ) -> ToolResult:
        if not isinstance(query, str) or not query.strip():
            return _failure("partial", "QUERY_REQUIRED", "A non-empty query is required.", ["query"])
        if not product_id:
            return _failure("partial", "PRODUCT_NOT_BOUND", "Locate the product before manual search.", ["current_product_id"])
        if not sku_id:
            return _failure("partial", "SKU_NOT_BOUND", "Locate the SKU before manual search.", ["current_sku_id"])
        key = (snapshot_id, product_id, sku_id)
        manual_id = self.bindings.get(key)
        if manual_id is None:
            return _failure("no_match", "MANUAL_BINDING_NOT_FOUND", "No frozen manual is approved for this session binding.")
        try:
            found = self.retriever.search(product_id, query.strip(), self.top_k)
            if not found or len({row["chunk_id"] for row in found}) != len(found):
                raise ValueError("empty or duplicate manual results")
            if any(row["manual_id"] != manual_id or row["product_id"] != product_id for row in found):
                raise ValueError("manual result escaped the session binding")
            chunks = self.chunks_by_product[product_id]
            results = []
            refs = []
            for row in found:
                if row["chunk_id"] not in chunks:
                    raise ValueError("retriever returned a chunk outside the frozen manual")
                source = chunks[row["chunk_id"]]
                result = {
                    "rank": row["rank"],
                    "section_id": row["chunk_id"],
                    "source_chunk_id": source["source_chunk_id"],
                    "section_title": source["section_title"],
                    "content": source["content"],
                    "score": row["score"],
                    "manual_id": manual_id,
                    "manual_version": source["manual_version"],
                    "chunk_sha256": source["chunk_sha256"],
                }
                results.append(result)
                refs.append({
                    "source": "frozen_manual_v1", "manual_id": manual_id,
                    "section_id": result["section_id"], "source_chunk_id": result["source_chunk_id"],
                    "chunk_sha256": result["chunk_sha256"],
                })
            return ToolResult(
                ok=True, tool=TOOL_NAME, status="found",
                data={
                    "snapshot_id": snapshot_id,
                    "product_id": product_id,
                    "sku_id": sku_id,
                    "manual_id": manual_id,
                    "query": query.strip(),
                    "results": results,
                    "retriever_config": dict(self.config),
                },
                evidence_refs=refs,
            )
        except Exception as exc:
            return _failure("tool_error", "MANUAL_SEARCH_FAILED", f"Manual search failed: {type(exc).__name__}: {exc}")
