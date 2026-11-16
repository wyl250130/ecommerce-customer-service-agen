#!/usr/bin/env python3
"""Evaluate a product-bound manual retriever on a JSONL Retriever Gold."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rag.retriever import ManualRetriever


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLD = ROOT / "manual_rag_retriever_gold_v4.jsonl"
DEFAULT_INDEX = ROOT / "data/rag_index/manual_v1_tfidf_char"
DEFAULT_OUTPUT = ROOT / "outputs/rag_retriever_baseline_v1_tfidf_char"
KS = (1, 2, 3, 5)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def classify_error(row: dict) -> list[str]:
    if not row["gold_chunk_ids"]:
        return ["gold_empty_manual_review"]
    labels = []
    if not row["hit_at_5"]:
        labels.append("no_gold_in_top5")
    elif not row["hit_at_3"]:
        labels.append("gold_ranked_4_or_5")
    elif not row["hit_at_1"]:
        labels.append("top1_ranking_error")
    if row.get("need_type", "").startswith("multi_section_") and row["evidence_recall_at_3"] < 1.0:
        labels.append("multi_section_incomplete_at_3")
    return labels


def evaluate(gold_path: Path, index_dir: Path, output_dir: Path, top_k: int = 5) -> dict:
    if top_k < max(KS):
        raise ValueError(f"top_k must be at least {max(KS)} for the required metrics")
    gold_rows = read_jsonl(gold_path)
    if not gold_rows:
        raise ValueError("Gold file is empty")
    retriever = ManualRetriever(index_dir)
    results = []
    seen_question_ids: set[str] = set()
    for gold in gold_rows:
        required = {"question_id", "manual_id", "product_id", "user_query", "gold_chunk_ids"}
        missing = required - set(gold)
        if missing:
            raise ValueError(f"{gold.get('question_id', '<unknown>')} missing {sorted(missing)}")
        if gold["question_id"] in seen_question_ids:
            raise ValueError(f"duplicate question_id: {gold['question_id']}")
        seen_question_ids.add(gold["question_id"])
        if gold["product_id"] not in retriever.product_indices:
            raise ValueError(f"unknown Gold product_id: {gold['product_id']}")
        product_rows = [
            retriever.metadata[int(index)]
            for index in retriever.product_indices[gold["product_id"]]
        ]
        valid_chunk_ids = {row["chunk_id"] for row in product_rows}
        invalid_gold_ids = set(gold["gold_chunk_ids"]) - valid_chunk_ids
        if invalid_gold_ids:
            raise ValueError(
                f"{gold['question_id']} has chunk ids outside the bound product: "
                f"{sorted(invalid_gold_ids)}"
            )
        valid_manual_ids = {row["manual_id"] for row in product_rows}
        if gold["manual_id"] not in valid_manual_ids:
            raise ValueError(
                f"{gold['question_id']} manual_id does not match product_id"
            )
        retrieved = retriever.search(gold["product_id"], gold["user_query"], top_k)
        if any(item["product_id"] != gold["product_id"] for item in retrieved):
            raise AssertionError("product filter leaked a chunk from another product")
        retrieved_ids = [item["chunk_id"] for item in retrieved]
        gold_ids = list(gold["gold_chunk_ids"])
        gold_set = set(gold_ids)
        acceptable_groups = gold.get("acceptable_gold_groups") or ([gold_ids] if gold_ids else [])
        for group in acceptable_groups:
            if not group or not set(group).issubset(gold_set):
                raise ValueError(f"{gold['question_id']} has invalid acceptable_gold_groups")
        row = {
            "question_id": gold["question_id"],
            "manual_id": gold["manual_id"],
            "product_id": gold["product_id"],
            "query": gold["user_query"],
            "gold_chunk_ids": gold_ids,
            "acceptable_gold_groups": acceptable_groups,
            "gold_empty": not gold_ids,
            "retrieved": retrieved,
        }
        for optional_field in ("query_id", "information_need_id", "need_type", "utterance_style", "split"):
            if optional_field in gold:
                row[optional_field] = gold[optional_field]
        best_groups_at_k = {}
        for k in KS:
            retrieved_at_k = set(retrieved_ids[:k])
            group_coverages = [len(set(group) & retrieved_at_k) / len(group) for group in acceptable_groups]
            row[f"evidence_recall_at_{k}"] = max(group_coverages) if group_coverages else None
            row[f"hit_at_{k}"] = (row[f"evidence_recall_at_{k}"] > 0) if gold_ids else None
            if acceptable_groups:
                best_index = max(range(len(acceptable_groups)), key=lambda index: group_coverages[index])
                best_groups_at_k[k] = acceptable_groups[best_index]
        first_gold_rank = next(
            (rank for rank, chunk_id in enumerate(retrieved_ids, 1) if chunk_id in gold_set), None
        )
        row["first_gold_rank"] = first_gold_rank
        row["reciprocal_rank"] = (1.0 / first_gold_rank) if first_gold_rank else (0.0 if gold_ids else None)
        row["best_acceptable_gold_group_at_3"] = best_groups_at_k.get(3, [])
        row["best_acceptable_gold_group_at_5"] = best_groups_at_k.get(5, [])
        row["missed_gold_chunks_at_3"] = [
            chunk for chunk in row["best_acceptable_gold_group_at_3"] if chunk not in retrieved_ids[:3]
        ]
        row["missed_gold_chunks_at_5"] = [
            chunk for chunk in row["best_acceptable_gold_group_at_5"] if chunk not in retrieved_ids[:5]
        ]
        row["error_types"] = classify_error(row)
        results.append(row)

    scored = [row for row in results if not row["gold_empty"]]
    empty = [row for row in results if row["gold_empty"]]
    metrics = {
        "run_name": output_dir.name,
        "backend_config": retriever.manifest["backend_config"],
        "gold_path": portable_path(gold_path),
        "gold_sha256": file_sha256(gold_path),
        "index_dir": portable_path(index_dir),
        "index_chunks_sha256": retriever.manifest["chunks_sha256"],
        "question_count": len(results),
        "scored_nonempty_count": len(scored),
        "gold_empty_count": len(empty),
    }
    for k in KS:
        metrics[f"recall_at_{k}"] = sum(row[f"hit_at_{k}"] for row in scored) / len(scored)
        metrics[f"evidence_recall_at_{k}"] = sum(
            row[f"evidence_recall_at_{k}"] for row in scored
        ) / len(scored)
        metrics[f"gold_chunk_recall_at_{k}"] = metrics[f"evidence_recall_at_{k}"]
    metrics["mrr"] = sum(row["reciprocal_rank"] for row in scored) / len(scored)
    metrics["full_evidence_recall_at_3"] = sum(
        row["evidence_recall_at_3"] == 1.0 for row in scored
    ) / len(scored)
    metrics["by_utterance_style"] = grouped_metrics(scored, "utterance_style")
    metrics["by_need_type"] = grouped_metrics(scored, "need_type")
    metrics["by_split"] = grouped_metrics(scored, "split")

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "retrieval_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (output_dir / "gold_empty_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in empty:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (output_dir / "retrieval_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_error_report(output_dir / "retrieval_error_report.md", metrics, results)
    return metrics


def grouped_metrics(rows: list[dict], field: str) -> dict:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        value = row.get(field)
        if value is not None:
            groups.setdefault(str(value), []).append(row)
    output = {}
    for value, group in sorted(groups.items()):
        item = {"count": len(group)}
        for k in KS:
            item[f"recall_at_{k}"] = sum(row[f"hit_at_{k}"] for row in group) / len(group)
            item[f"gold_chunk_recall_at_{k}"] = sum(
                row[f"evidence_recall_at_{k}"] for row in group
            ) / len(group)
        item["mrr"] = sum(row["reciprocal_rank"] for row in group) / len(group)
        item["full_evidence_recall_at_3"] = sum(
            row["evidence_recall_at_3"] == 1.0 for row in group
        ) / len(group)
        output[value] = item
    return output


def write_error_report(path: Path, metrics: dict, results: list[dict]) -> None:
    failures = [row for row in results if row["error_types"]]
    lines = [
        "# Manual RAG Retriever Baseline 错误分析",
        "",
        f"- Backend：`{metrics['backend_config']['backend']}`",
        f"- 题目总数：{metrics['question_count']}",
        f"- 计分题：{metrics['scored_nonempty_count']}",
        f"- Gold-empty：{metrics['gold_empty_count']}（不进入主指标）",
        f"- Recall@3：{metrics['recall_at_3']:.4f}",
        f"- Gold Chunk Recall@3：{metrics['gold_chunk_recall_at_3']:.4f}",
        f"- MRR：{metrics['mrr']:.4f}",
        "",
        "说明：本报告列出 Top-1 排序失败、Top-3 多证据不完整、Top-5 未召回，以及 Gold-empty 人工观察项。",
        "",
    ]
    for row in failures:
        lines.extend([
            f"## {row['question_id']}",
            "",
            "### Query",
            "",
            row["query"],
            "",
            "### Gold",
            "",
        ])
        if row["gold_chunk_ids"]:
            lines.extend(f"- `{chunk_id}`" for chunk_id in row["gold_chunk_ids"])
            if row.get("acceptable_gold_groups"):
                rendered = [" + ".join(group) for group in row["acceptable_gold_groups"]]
                lines.append(f"- 可接受证据组：{'；'.join(rendered)}")
        else:
            lines.append("- （空；本题不进入主指标）")
        lines.extend(["", "### Top-5", ""])
        for item in row["retrieved"][:5]:
            lines.append(
                f"{item['rank']}. `{item['chunk_id']}` / {item['section_title']} — {item['score']:.6f}"
            )
        lines.extend([
            "",
            "### Missed Gold at Top-3",
            "",
        ])
        if row["missed_gold_chunks_at_3"]:
            lines.extend(f"- `{chunk_id}`" for chunk_id in row["missed_gold_chunks_at_3"])
        else:
            lines.append("- 无")
        lines.extend([
            "",
            "### Error Type",
            "",
        ])
        lines.extend(f"- `{label}`" for label in row["error_types"])
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate(args.gold, args.index_dir, args.output_dir, args.top_k)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
