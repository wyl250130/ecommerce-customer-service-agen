#!/usr/bin/env python3
"""Run the Direct Cross-Encoder over frozen handwritten v2 Retriever Gold."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from rag.direct_cross_encoder import (
    MODEL_NAME, MODEL_REVISION, DirectCrossEncoderRetriever, HuggingFaceCrossEncoder,
    normalize_chunk, passage_text,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLD = ROOT / "data/processed/retriever_eval/manual_v1_handwritten_v2_compiled.jsonl"
DEFAULT_CHUNKS = ROOT / "data/processed/product_manuals/frozen_v1/product_manual_chunks_frozen_v1.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs/retriever_handwritten_v2_final/direct_bge_reranker_v2_m3_content_template"
EXPECTED_GOLD_SHA256 = "5ff9cd7c476127ddcb9a9ea23ca45fe304b67e13cdef5ddff0067c520f32b27b"
EXPECTED_CHUNKS_SHA256 = "3d2c5585088d355cd75120a03888a20709fce4f087ac03b6a6cf52891756500a"
KS = (1, 2, 3, 5)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def grouped_metrics(rows: list[dict], field: str) -> dict:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        if row.get(field) is not None:
            groups.setdefault(str(row[field]), []).append(row)
    output = {}
    for value, group in sorted(groups.items()):
        output[value] = _metrics_for(group)
    return output


def _metrics_for(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0}
    result = {"count": len(rows)}
    for k in KS:
        result[f"recall_at_{k}"] = sum(row[f"hit_at_{k}"] for row in rows) / len(rows)
        result[f"gold_chunk_recall_at_{k}"] = sum(
            row[f"evidence_recall_at_{k}"] for row in rows
        ) / len(rows)
    result["mrr"] = sum(row["reciprocal_rank"] for row in rows) / len(rows)
    result["full_evidence_recall_at_3"] = sum(
        row["evidence_recall_at_3"] == 1.0 for row in rows
    ) / len(rows)
    return result


def describe(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    ordered = sorted(float(value) for value in values)
    def percentile(p: float) -> float:
        index = (len(ordered) - 1) * p
        low = int(index)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] * (high - index) + ordered[high] * (index - low)
    return {
        "count": len(ordered), "mean": statistics.fmean(ordered),
        "p50": percentile(0.50), "p95": percentile(0.95),
        "min": ordered[0], "max": ordered[-1],
    }


def build_result(gold: dict, retrieved: list[dict]) -> dict:
    gold_ids = list(gold["gold_chunk_ids"])
    gold_set = set(gold_ids)
    acceptable = gold.get("acceptable_gold_groups") or ([gold_ids] if gold_ids else [])
    for group in acceptable:
        if not group or not set(group).issubset(gold_set):
            raise ValueError(f"{gold['question_id']} has invalid acceptable_gold_groups")
    retrieved_ids = [item["chunk_id"] for item in retrieved]
    row = {
        "question_id": gold["question_id"], "manual_id": gold["manual_id"],
        "product_id": gold["product_id"], "query": gold["user_query"],
        "gold_chunk_ids": gold_ids, "acceptable_gold_groups": acceptable,
        "gold_empty": not gold_ids, "retrieved": retrieved,
    }
    for field in ("query_id", "information_need_id", "need_type", "utterance_style", "split"):
        if field in gold:
            row[field] = gold[field]
    coverage = {}
    for k in KS:
        hit = set(retrieved_ids[:k])
        group_scores = [len(set(group) & hit) / len(group) for group in acceptable]
        coverage[k] = max(group_scores) if group_scores else None
        row[f"evidence_recall_at_{k}"] = coverage[k]
        row[f"hit_at_{k}"] = (coverage[k] > 0) if gold_ids else None
    first_gold = next((rank for rank, chunk_id in enumerate(retrieved_ids, 1) if chunk_id in gold_set), None)
    row["first_gold_rank"] = first_gold
    row["reciprocal_rank"] = 1.0 / first_gold if first_gold else (0.0 if gold_ids else None)
    best = max(acceptable, key=lambda group: len(set(group) & set(retrieved_ids[:3])) / len(group)) if acceptable else []
    row["best_acceptable_gold_group_at_3"] = best
    row["best_acceptable_gold_group_at_5"] = max(
        acceptable, key=lambda group: len(set(group) & set(retrieved_ids[:5])) / len(group)
    ) if acceptable else []
    row["missed_gold_chunks_at_3"] = [chunk for chunk in best if chunk not in retrieved_ids[:3]]
    row["missed_gold_chunks_at_5"] = [
        chunk for chunk in row["best_acceptable_gold_group_at_5"] if chunk not in retrieved_ids[:5]
    ]
    errors = []
    if row["gold_empty"]:
        errors = ["gold_empty_manual_review"]
    elif not row["hit_at_5"]:
        errors.append("no_gold_in_top5")
    elif not row["hit_at_3"]:
        errors.append("gold_ranked_4_or_5")
    elif not row["hit_at_1"]:
        errors.append("top1_ranking_error")
    if str(gold.get("need_type", "")).startswith("multi_section_") and row["evidence_recall_at_3"] < 1.0:
        errors.append("multi_section_incomplete_at_3")
    row["error_types"] = errors
    return row


def no_answer_analysis(rows: list[dict]) -> dict:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError:
        roc_auc_score = average_precision_score = None
    labels = [0 if row["gold_empty"] else 1 for row in rows]
    scores = [row["max_score"] for row in rows]
    need_groups: dict[str, list[dict]] = {}
    for row in rows:
        need_groups.setdefault(row.get("information_need_id", row["question_id"]), []).append(row)
    need_labels = [0 if group[0]["gold_empty"] else 1 for group in need_groups.values()]
    need_scores = [statistics.fmean(row["max_score"] for row in group) for group in need_groups.values()]
    def aucs(y, pred):
        if roc_auc_score is None or len(set(y)) < 2:
            return {"roc_auc": None, "pr_auc": None}
        return {"roc_auc": float(roc_auc_score(y, pred)), "pr_auc": float(average_precision_score(y, pred))}
    def by_group(field: str) -> dict:
        values: dict[str, list[dict]] = {}
        for row in rows:
            values.setdefault(str(row.get(field, "unknown")), []).append(row)
        return {key: {
            "answerable": describe([x["max_score"] for x in items if not x["gold_empty"]]),
            "no_answer": describe([x["max_score"] for x in items if x["gold_empty"]]),
        } for key, items in sorted(values.items())}
    return {
        "score_definition": "max raw logit among six product-bound chunks; not a calibrated probability",
        "query_level": {
            "answerable_max_score": describe([x["max_score"] for x in rows if not x["gold_empty"]]),
            "no_answer_max_score": describe([x["max_score"] for x in rows if x["gold_empty"]]),
            **aucs(labels, scores),
        },
        "information_need_level": {
            "need_count": len(need_groups),
            "answerable_need_count": sum(need_labels),
            "no_answer_need_count": len(need_labels) - sum(need_labels),
            "answerable_mean_max_score": describe([s for y, s in zip(need_labels, need_scores) if y]),
            "no_answer_mean_max_score": describe([s for y, s in zip(need_labels, need_scores) if not y]),
            **aucs(need_labels, need_scores),
        },
        "by_utterance_style": by_group("utterance_style"),
        "by_manual": by_group("manual_id"),
        "by_split": by_group("split"),
        "high_score_no_answer": sorted(
            [{"question_id": x["question_id"], "information_need_id": x.get("information_need_id"),
              "query": x["query"], "max_score": x["max_score"], "score_margin": x["score_margin"],
              "top1_chunk_id": x["retrieved"][0]["chunk_id"]} for x in rows if x["gold_empty"]],
            key=lambda item: -item["max_score"],
        ),
    }


def evaluate(gold_path: Path, chunks_path: Path, output_dir: Path, model_name: str = MODEL_NAME,
             revision: str = MODEL_REVISION, max_length: int = 512, batch_size: int = 6,
             device: str = "auto", passage_template: str = "full", cache_dir: Path | None = None,
             local_files_only: bool = False, warmup: int = 2) -> dict:
    gold_hash, chunks_hash = sha256(gold_path), sha256(chunks_path)
    if gold_hash != EXPECTED_GOLD_SHA256:
        raise ValueError(f"Gold hash differs from frozen baseline: {gold_hash}")
    if chunks_hash != EXPECTED_CHUNKS_SHA256:
        raise ValueError(f"chunks hash differs from frozen baseline: {chunks_hash}")
    gold_rows = read_jsonl(gold_path)
    chunks = [normalize_chunk(row) for row in read_jsonl(chunks_path)]
    if len(chunks) != 48:
        raise ValueError(f"expected 48 frozen chunks, found {len(chunks)}")
    if len(gold_rows) != 800:
        raise ValueError(f"expected 800 frozen query variants, found {len(gold_rows)}")
    started = datetime.now(timezone.utc).isoformat()
    load_start = time.perf_counter()
    scorer = HuggingFaceCrossEncoder(
        model_name=model_name, revision=revision, max_length=max_length,
        batch_size=batch_size, device=device, cache_dir=str(cache_dir) if cache_dir else None,
        local_files_only=local_files_only,
    )
    scorer.synchronize()
    model_load_seconds = time.perf_counter() - load_start
    retriever = DirectCrossEncoderRetriever(chunks, scorer.score_pairs, passage_template, 6)
    # Warm up on a real query but keep warm-up outside query latency.
    warm_pairs = retriever.candidate_pairs(gold_rows[0]["product_id"], gold_rows[0]["user_query"])[1]
    for _ in range(warmup):
        scorer.score_pairs(warm_pairs)
    scorer.synchronize()
    results, latencies = [], []
    evaluation_started = time.perf_counter()
    lengths, truncated = [], 0
    seen_ids: set[str] = set()
    for gold in gold_rows:
        required = {"question_id", "manual_id", "product_id", "user_query", "gold_chunk_ids"}
        missing = required - set(gold)
        if missing:
            raise ValueError(f"{gold.get('question_id', '<unknown>')} missing {sorted(missing)}")
        if gold["question_id"] in seen_ids:
            raise ValueError(f"duplicate question_id: {gold['question_id']}")
        seen_ids.add(gold["question_id"])
        if gold["product_id"] not in retriever.by_product:
            raise ValueError(f"unknown Gold product_id: {gold['product_id']}")
        candidates, pairs = retriever.candidate_pairs(gold["product_id"], gold["user_query"])
        candidate_ids = {row["chunk_id"] for row in candidates}
        if not set(gold["gold_chunk_ids"]).issubset(candidate_ids):
            raise ValueError(f"{gold['question_id']} contains Gold outside bound product")
        if gold["manual_id"] not in {row["manual_id"] for row in candidates}:
            raise ValueError(f"{gold['question_id']} manual_id/product_id mismatch")
        pair_lengths = [scorer.token_length(query, passage) for query, passage in pairs]
        lengths.extend(pair_lengths)
        truncated += sum(length > max_length for length in pair_lengths)
        scorer.synchronize()
        before = time.perf_counter()
        retrieved = retriever.search(gold["product_id"], gold["user_query"], top_k=6)
        scorer.synchronize()
        latencies.append(time.perf_counter() - before)
        if len(retrieved) != 6 or len({x["chunk_id"] for x in retrieved}) != 6:
            raise AssertionError("each query must return all six unique candidates")
        row = build_result(gold, retrieved)
        row["max_score"] = retrieved[0]["score"]
        row["score_margin"] = retrieved[0]["score"] - retrieved[1]["score"]
        results.append(row)
        if len(results) % 50 == 0:
            elapsed = time.perf_counter() - evaluation_started
            print(f"Scored {len(results)}/{len(gold_rows)} queries in {elapsed:.1f}s", flush=True)

    scored = [row for row in results if not row["gold_empty"]]
    empty = [row for row in results if row["gold_empty"]]
    metrics = {
        "run_name": output_dir.name, "backend": "direct_cross_encoder",
        "model_name": model_name, "model_revision": revision,
        "passage_template": passage_template, "max_length": max_length,
        "candidate_count": 6, "question_count": len(results),
        "scored_nonempty_count": len(scored), "gold_empty_count": len(empty),
        "gold_path": portable(gold_path), "gold_sha256": gold_hash,
        "chunks_path": portable(chunks_path), "chunks_sha256": chunks_hash,
        "index_chunks_sha256": chunks_hash,
    }
    metrics.update(_metrics_for(scored))
    metrics["by_utterance_style"] = grouped_metrics(scored, "utterance_style")
    metrics["by_need_type"] = grouped_metrics(scored, "need_type")
    metrics["by_split"] = grouped_metrics(scored, "split")
    metrics["by_manual"] = grouped_metrics(scored, "manual_id")
    metrics["by_product"] = grouped_metrics(scored, "product_id")
    latency = {
        "device": scorer.device, "dtype": "float32", "batch_size_pairs": batch_size,
        "candidates_per_query": 6, "max_length": max_length, "warmup_queries": warmup,
        "model_load_seconds": model_load_seconds,
        "single_query_latency_seconds": describe(latencies),
        "queries_per_second": len(latencies) / sum(latencies) if sum(latencies) else None,
        "pair_token_length": describe(lengths), "pair_count": len(lengths),
        "truncated_pair_count": truncated,
        "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024)),
        "hardware": platform.platform(), "processor": platform.processor(),
        "note": "Latency covers one query with six candidate pairs, including tokenization; model loading and warmup are separate.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    def write_jsonl(path: Path, rows: list[dict]):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_jsonl(output_dir / "retrieval_results.jsonl", results)
    write_jsonl(output_dir / "gold_empty_results.jsonl", empty)
    (output_dir / "retrieval_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "latency_metrics.json").write_text(json.dumps(latency, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    analysis = no_answer_analysis(results)
    (output_dir / "no_answer_analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_error_report(output_dir / "retrieval_error_report.md", metrics, results)
    import numpy, torch, transformers, sklearn
    package_versions = {
        "python": platform.python_version(), "numpy": numpy.__version__, "torch": torch.__version__,
        "transformers": transformers.__version__, "scikit_learn": sklearn.__version__,
    }
    manifest = {
        "run_name": output_dir.name, "model_name": model_name, "model_revision": revision,
        "mode": "direct_cross_encoder_all_product_chunks", "candidate_count_expected": 6,
        "passage_format": "商品：{product_name}\\n章节：{section_title}\\n内容：{content}" if passage_template == "full" else "content_only",
        "max_length": max_length, "batch_size_pairs": batch_size, "dtype": "float32",
        "device": scorer.device, "gold_path": portable(gold_path), "gold_sha256": gold_hash,
        "chunks_path": portable(chunks_path), "chunks_sha256": chunks_hash,
        "seed": 42, "package_versions": package_versions, "git_commit": git_commit(),
        "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
        "query_instruction": None, "truncation": "only_second; query preserved",
        "artifacts": {},
    }
    for name in ("retrieval_results.jsonl", "gold_empty_results.jsonl", "retrieval_metrics.json", "latency_metrics.json", "no_answer_analysis.json", "retrieval_error_report.md"):
        manifest["artifacts"][name] = sha256(output_dir / name)
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"metrics": metrics, "latency": latency, "no_answer_analysis": analysis, "manifest": manifest}


def write_error_report(path: Path, metrics: dict, rows: list[dict]) -> None:
    failures = [row for row in rows if row["error_types"]]
    lines = ["# Direct Cross-Encoder 错误分析", "",
             f"- 计分题：{metrics['scored_nonempty_count']}",
             f"- Gold-empty：{metrics['gold_empty_count']}（不进入主指标）",
             f"- Recall@1：{metrics['recall_at_1']:.4f}",
             f"- Gold Chunk Recall@3：{metrics['gold_chunk_recall_at_3']:.4f}",
             f"- Full Evidence Recall@3：{metrics['full_evidence_recall_at_3']:.4f}", ""]
    for row in failures:
        lines.extend([f"## {row['question_id']}", "", row["query"], "", f"- Gold: {row['gold_chunk_ids']}", "", "### Top 6", ""])
        lines.extend(f"{x['rank']}. `{x['chunk_id']}` / {x['section_title']} — {x['score']:.6f}" for x in row["retrieved"])
        lines.extend(["", f"- Errors: {', '.join(row['error_types'])}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--passage-template", choices=("full", "content_only"), default="full")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--top-k", type=int, choices=(6,), default=6, help="Direct mode always scores and saves all six product chunks")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/huggingface")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(args.gold, args.chunks, args.output_dir, args.model_name,
                      args.revision, args.max_length, args.batch_size, args.device,
                      args.passage_template, args.cache_dir, args.local_files_only, args.warmup)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
