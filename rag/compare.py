#!/usr/bin/env python3
"""Compare TF-IDF and BGE retrieval runs question by question."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TFIDF = ROOT / "outputs/rag_retriever_baseline_v1_tfidf_char"
DEFAULT_BGE = ROOT / "outputs/rag_retriever_baseline_v1_bge_small_zh"
DEFAULT_OUTPUT = ROOT / "outputs/rag_retriever_comparison_v1"
METRICS = (
    "recall_at_1", "recall_at_3", "recall_at_5",
    "gold_chunk_recall_at_1", "gold_chunk_recall_at_3", "gold_chunk_recall_at_5",
    "full_evidence_recall_at_3", "mrr",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def classify(tfidf: dict, bge: dict) -> str:
    if tfidf["gold_empty"]:
        return "unscored_gold_empty"
    # Coverage is the primary objective for multi-evidence questions; first-Gold
    # rank breaks ties. This gives one deterministic three-way label per query.
    tfidf_key = (tfidf["evidence_recall_at_3"], tfidf["reciprocal_rank"])
    bge_key = (bge["evidence_recall_at_3"], bge["reciprocal_rank"])
    if bge_key > tfidf_key:
        return "improved"
    if bge_key < tfidf_key:
        return "regressed"
    return "unchanged"


def subset_metrics(rows: list[dict], system: str) -> dict:
    if not rows:
        return {"count": 0}
    prefix = system
    result = {"count": len(rows)}
    for k in (1, 3, 5):
        result[f"recall_at_{k}"] = sum(row[prefix][f"hit_at_{k}"] for row in rows) / len(rows)
        result[f"gold_chunk_recall_at_{k}"] = sum(
            row[prefix][f"evidence_recall_at_{k}"] for row in rows
        ) / len(rows)
    result["full_evidence_recall_at_3"] = sum(
        row[prefix]["evidence_recall_at_3"] == 1.0 for row in rows
    ) / len(rows)
    result["mrr"] = sum(row[prefix]["reciprocal_rank"] for row in rows) / len(rows)
    return result


def compare(tfidf_dir: Path, bge_dir: Path, output_dir: Path) -> dict:
    tfidf_metrics = json.loads((tfidf_dir / "retrieval_metrics.json").read_text(encoding="utf-8"))
    bge_metrics = json.loads((bge_dir / "retrieval_metrics.json").read_text(encoding="utf-8"))
    tfidf_rows = {row["question_id"]: row for row in read_jsonl(tfidf_dir / "retrieval_results.jsonl")}
    bge_rows = {row["question_id"]: row for row in read_jsonl(bge_dir / "retrieval_results.jsonl")}
    if set(tfidf_rows) != set(bge_rows):
        raise ValueError("the two runs contain different question ids")
    for field in ("gold_sha256", "index_chunks_sha256", "question_count"):
        if tfidf_metrics[field] != bge_metrics[field]:
            raise ValueError(f"unfair comparison: {field} differs")

    rows = []
    for question_id in sorted(tfidf_rows):
        left = tfidf_rows[question_id]
        right = bge_rows[question_id]
        if left["gold_chunk_ids"] != right["gold_chunk_ids"]:
            raise ValueError(f"Gold differs for {question_id}")
        if left.get("acceptable_gold_groups") != right.get("acceptable_gold_groups"):
            raise ValueError(f"acceptable Gold groups differ for {question_id}")
        change = classify(left, right)
        row = {
            "question_id": question_id,
            "product_id": left["product_id"],
            "query": left["query"],
            "gold_chunk_ids": left["gold_chunk_ids"],
            "gold_count": len(left["gold_chunk_ids"]),
            "need_type": left.get("need_type"),
            "acceptable_gold_groups": left.get("acceptable_gold_groups"),
            "change": change,
            "tfidf": {
                "first_gold_rank": left["first_gold_rank"],
                "reciprocal_rank": left["reciprocal_rank"],
                "hit_at_1": left["hit_at_1"],
                "hit_at_3": left["hit_at_3"],
                "hit_at_5": left["hit_at_5"],
                "evidence_recall_at_1": left["evidence_recall_at_1"],
                "evidence_recall_at_3": left["evidence_recall_at_3"],
                "evidence_recall_at_5": left["evidence_recall_at_5"],
                "top3": [item["chunk_id"] for item in left["retrieved"][:3]],
            },
            "bge": {
                "first_gold_rank": right["first_gold_rank"],
                "reciprocal_rank": right["reciprocal_rank"],
                "hit_at_1": right["hit_at_1"],
                "hit_at_3": right["hit_at_3"],
                "hit_at_5": right["hit_at_5"],
                "evidence_recall_at_1": right["evidence_recall_at_1"],
                "evidence_recall_at_3": right["evidence_recall_at_3"],
                "evidence_recall_at_5": right["evidence_recall_at_5"],
                "top3": [item["chunk_id"] for item in right["retrieved"][:3]],
            },
        }
        rows.append(row)

    scored = [row for row in rows if row["gold_count"] > 0]
    single = [row for row in scored if row.get("need_type") == "single_section"]
    multi = [row for row in scored if str(row.get("need_type", "")).startswith("multi_section_")]
    changes = Counter(row["change"] for row in rows)
    aggregate = {}
    for metric in METRICS:
        aggregate[metric] = {
            "tfidf": tfidf_metrics[metric],
            "bge": bge_metrics[metric],
            "delta": bge_metrics[metric] - tfidf_metrics[metric],
        }
    summary = {
        "comparison": "tfidf_char_vs_BAAI_bge_small_zh_v1.5",
        "fairness_checks": {
            "same_gold_sha256": True,
            "same_chunks_sha256": True,
            "same_question_count": True,
            "same_product_filter_and_evaluator": True,
        },
        "question_count": len(rows),
        "scored_nonempty_count": len(scored),
        "gold_empty_count": len(rows) - len(scored),
        "change_counts": dict(sorted(changes.items())),
        "classification_policy": "Compare (Gold Chunk Recall@3, reciprocal rank) lexicographically; coverage has priority.",
        "aggregate": aggregate,
        "single_gold": {
            "tfidf": subset_metrics(single, "tfidf"),
            "bge": subset_metrics(single, "bge"),
        },
        "multi_gold": {
            "tfidf": subset_metrics(multi, "tfidf"),
            "bge": subset_metrics(multi, "bge"),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "retriever_comparison.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_report(output_dir / "comparison_report.md", summary, rows)
    return summary


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_report(path: Path, summary: dict, rows: list[dict]) -> None:
    labels = {
        "recall_at_1": "Recall@1", "recall_at_3": "Recall@3", "recall_at_5": "Recall@5",
        "gold_chunk_recall_at_1": "Gold Chunk Recall@1",
        "gold_chunk_recall_at_3": "Gold Chunk Recall@3",
        "gold_chunk_recall_at_5": "Gold Chunk Recall@5",
        "full_evidence_recall_at_3": "Full Evidence Recall@3", "mrr": "MRR",
    }
    lines = [
        "# TF-IDF vs BGE Retriever 对照报告", "",
        "## 公平性", "",
        "两次实验使用完全相同的说明书 chunks、Retriever Gold、product_id filter、Top-K 和 Evaluator；唯一主要变量是向量表示。", "",
        "## 聚合指标", "",
        "| Metric | TF-IDF | BGE-small-zh | Delta |", "|---|---:|---:|---:|",
    ]
    for key in METRICS:
        item = summary["aggregate"][key]
        if key == "mrr":
            values = (f"{item['tfidf']:.4f}", f"{item['bge']:.4f}", f"{item['delta']:+.4f}")
        else:
            values = (pct(item["tfidf"]), pct(item["bge"]), f"{item['delta'] * 100:+.2f} pp")
        lines.append(f"| {labels[key]} | {values[0]} | {values[1]} | {values[2]} |")
    lines.extend([
        "", "## 逐题变化", "",
        f"- Improved：{summary['change_counts'].get('improved', 0)}",
        f"- Unchanged：{summary['change_counts'].get('unchanged', 0)}",
        f"- Regressed：{summary['change_counts'].get('regressed', 0)}",
        f"- Gold-empty（不计分）：{summary['change_counts'].get('unscored_gold_empty', 0)}",
        "",
        "分类先比较 Top-3 Gold chunk 覆盖率，覆盖率相同时再比较第一个 Gold 的排名。", "",
    ])
    for change in ("improved", "regressed"):
        title = "改善题" if change == "improved" else "退化题"
        lines.extend([f"## {title}", ""])
        selected = [row for row in rows if row["change"] == change]
        if not selected:
            lines.append("无。")
            lines.append("")
            continue
        for row in selected:
            lines.extend([
                f"### {row['question_id']}", "", row["query"], "",
                f"- Gold：{', '.join(row['gold_chunk_ids'])}",
                f"- TF-IDF：rank={row['tfidf']['first_gold_rank']}，Top3={row['tfidf']['top3']}，coverage={row['tfidf']['evidence_recall_at_3']}",
                f"- BGE：rank={row['bge']['first_gold_rank']}，Top3={row['bge']['top3']}，coverage={row['bge']['evidence_recall_at_3']}",
                "",
            ])
    lines.extend([
        "## 结论", "",
        "请结合当前数据版本的逐题结果解释差异；不要仅凭模型类型替换默认 Retriever。后续优化应作为独立实验运行。", "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfidf-dir", type=Path, default=DEFAULT_TFIDF)
    parser.add_argument("--bge-dir", type=Path, default=DEFAULT_BGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(compare(args.tfidf_dir, args.bge_dir, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
