"""Score retrieval results against the existing human-authored frozen-manual Gold.

Predictions JSONL: {"query_id": "...", "retrieved_chunk_ids": ["MANUAL::section", ...]}.
Only the frozen-product queries have human labels; legacy category bindings do not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.unified_manuals import DEFAULT_CORPUS, UnifiedManualCorpus


def evaluate(corpus: UnifiedManualCorpus, *, split: str = "dev",
             predictions: Path | None = None, top_k: int = 5) -> dict:
    if top_k < 5:
        raise ValueError("top_k must be at least 5 for @5 metrics")
    with (corpus.directory / "human_queries.jsonl").open(encoding="utf-8") as source:
        queries = [row for line in source if line.strip()
                   for row in [json.loads(line)] if row["split"] == split]
    if not queries:
        raise ValueError(f"no queries for split {split}")
    provided = None
    if predictions is not None:
        with predictions.open(encoding="utf-8") as source:
            rows = [json.loads(line) for line in source if line.strip()]
        provided = {row["query_id"]: row["retrieved_chunk_ids"] for row in rows}
        if len(provided) != len(rows):
            raise ValueError("duplicate prediction query_id")
    summary = {"split": split, "queries": len(queries), "answerable": 0,
               "no_answer": 0, "hit_at_1": 0, "hit_at_3": 0,
               "hit_at_5": 0, "full_evidence_at_5": 0, "mrr_sum": 0.0}
    for row in queries:
        snapshot_id, product_id, sku_id = json.loads(row["binding_key"])
        allowed = {item["chunk_id"] for item in corpus.candidates(snapshot_id, product_id, sku_id)}
        if provided is None:
            retrieved = [item["chunk_id"] for item in corpus.search(
                snapshot_id, product_id, sku_id, row["user_query"], top_k=top_k)]
        else:
            if row["query_id"] not in provided:
                raise ValueError(f"missing prediction: {row['query_id']}")
            retrieved = provided[row["query_id"]][:top_k]
        if len(retrieved) != len(set(retrieved)) or not set(retrieved).issubset(allowed):
            raise ValueError(f"retrieval escaped binding or duplicated a chunk: {row['query_id']}")
        gold = set(row["gold_chunk_ids"])
        if not gold:
            summary["no_answer"] += 1
            continue
        summary["answerable"] += 1
        for k in (1, 3, 5):
            summary[f"hit_at_{k}"] += bool(gold & set(retrieved[:k]))
        summary["full_evidence_at_5"] += gold.issubset(set(retrieved[:5]))
        rank = next((i for i, chunk_id in enumerate(retrieved, 1) if chunk_id in gold), None)
        if rank:
            summary["mrr_sum"] += 1 / rank
    count = summary["answerable"]
    return {"split": split, "queries": summary["queries"], "answerable": count,
            "no_answer_excluded": summary["no_answer"],
            "hit_at_1": summary["hit_at_1"] / count,
            "hit_at_3": summary["hit_at_3"] / count,
            "hit_at_5": summary["hit_at_5"] / count,
            "full_evidence_at_5": summary["full_evidence_at_5"] / count,
            "mrr": summary["mrr_sum"] / count,
            "ranker": "char_bm25_baseline" if provided is None else str(predictions)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    report = evaluate(UnifiedManualCorpus(args.corpus), split=args.split,
                      predictions=args.predictions, top_k=args.top_k)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
