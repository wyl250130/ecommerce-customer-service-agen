"""Exact-binding candidate access for the mixed manual benchmark corpus.

Pass `candidates()` to any retriever under comparison. No product-name matching
or cross-snapshot fallback is performed at query time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "data/processed/product_manuals/unified_v2"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding_key(snapshot_id: str, product_id: str, sku_id: str) -> str:
    if not all(isinstance(value, str) and value for value in (snapshot_id, product_id, sku_id)):
        raise ValueError("snapshot_id, product_id and sku_id are required")
    return json.dumps([snapshot_id, product_id, sku_id], ensure_ascii=False, separators=(",", ":"))


class UnifiedManualCorpus:
    def __init__(self, directory: Path | str = DEFAULT_CORPUS):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("version") != "unified_manual_corpus_v2":
            raise ValueError("unsupported corpus version")
        for filename, expected in self.manifest["output_sha256"].items():
            if _hash(self.directory / filename) != expected:
                raise ValueError(f"corpus file changed: {filename}")
        self.chunks = _read_jsonl(self.directory / "chunks.jsonl")
        self.bindings = _read_jsonl(self.directory / "bindings.jsonl")
        self.chunks_by_manual: dict[str, list[dict]] = {}
        self.bindings_by_key: dict[str, list[dict]] = {}
        for row in self.chunks:
            self.chunks_by_manual.setdefault(row["manual_id"], []).append(row)
        for row in self.bindings:
            if row["manual_id"] not in self.chunks_by_manual:
                raise ValueError(f"binding has no chunks: {row['manual_id']}")
            self.bindings_by_key.setdefault(row["binding_key"], []).append(row)

    def candidates(self, snapshot_id: str, product_id: str, sku_id: str,
                   *, include_legacy_overlap: bool = False) -> list[dict]:
        rows = self.bindings_by_key.get(binding_key(snapshot_id, product_id, sku_id), [])
        if not include_legacy_overlap and any(row["source_tier"] == "frozen_product" for row in rows):
            rows = [row for row in rows if row["source_tier"] == "frozen_product"]
        return [dict(chunk, snapshot_id=snapshot_id, product_id=product_id, sku_id=sku_id,
                     binding_source=row["match_basis"])
                for row in rows for chunk in self.chunks_by_manual[row["manual_id"]]]

    def search(self, snapshot_id: str, product_id: str, sku_id: str, query: str,
               *, top_k: int = 5, include_legacy_overlap: bool = False) -> list[dict]:
        """Small dependency-free char BM25 baseline; candidates() supports other rankers."""
        if not isinstance(query, str) or not query.strip() or top_k < 1:
            raise ValueError("query must be nonempty and top_k positive")
        rows = self.candidates(snapshot_id, product_id, sku_id,
                               include_legacy_overlap=include_legacy_overlap)
        if not rows:
            return []
        def grams(value: str) -> Counter[str]:
            compact = "".join(value.lower().split())
            return Counter(compact[i:i + n] for n in (2, 3)
                           for i in range(max(0, len(compact) - n + 1)))
        docs = [grams(row["section_title"] + row["content"]) for row in rows]
        terms = grams(query)
        df = Counter(term for doc in docs for term in doc)
        lengths = [sum(doc.values()) for doc in docs]
        avg_len = sum(lengths) / len(lengths) or 1
        scored = []
        for row, doc, length in zip(rows, docs, lengths):
            score = 0.0
            for term, count in terms.items():
                freq = doc.get(term, 0)
                if freq:
                    idf = math.log(1 + (len(rows) - df[term] + 0.5) / (df[term] + 0.5))
                    score += count * idf * freq * 2.2 / (freq + 1.2 * (0.25 + 0.75 * length / avg_len))
            scored.append(dict(row, score=score))
        return sorted(scored, key=lambda row: (-row["score"], row["chunk_id"]))[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--product-id", required=True)
    parser.add_argument("--sku-id", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--include-legacy-overlap", action="store_true")
    args = parser.parse_args()
    corpus = UnifiedManualCorpus(args.corpus)
    results = corpus.search(args.snapshot_id, args.product_id, args.sku_id, args.query,
                            top_k=args.top_k,
                            include_legacy_overlap=args.include_legacy_overlap)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
