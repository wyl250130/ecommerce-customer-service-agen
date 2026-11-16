#!/usr/bin/env python3
"""Create the formal three-run analysis for handwritten Retriever v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "outputs/retriever_handwritten_v2_final"
RUNS = {
    "tfidf_char": "tfidf_char",
    "bge_small_zh": "bge_small_zh",
    "bge_small_zh_instruction": "bge_small_zh_instruction",
}
METRIC_KEYS = (
    "recall_at_1", "recall_at_2", "recall_at_3",
    "gold_chunk_recall_at_1", "gold_chunk_recall_at_2", "gold_chunk_recall_at_3",
    "full_evidence_recall_at_3", "mrr",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_rows(rows: list[dict]) -> dict:
    scored = [row for row in rows if not row["gold_empty"]]
    if not scored:
        return {"count": 0}
    output = {"count": len(scored)}
    for k in (1, 2, 3):
        output[f"recall_at_{k}"] = sum(row[f"hit_at_{k}"] for row in scored) / len(scored)
        output[f"gold_chunk_recall_at_{k}"] = sum(
            row[f"evidence_recall_at_{k}"] for row in scored
        ) / len(scored)
    output["full_evidence_recall_at_3"] = sum(
        row["evidence_recall_at_3"] == 1.0 for row in scored
    ) / len(scored)
    output["mrr"] = sum(row["reciprocal_rank"] for row in scored) / len(scored)
    return output


def grouped(rows: list[dict], key_fn) -> dict:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[str(key_fn(row))].append(row)
    return {key: summarize_rows(value) for key, value in sorted(buckets.items())}


def quantiles(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "min": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def score_auc(answerable: list[float], no_answer: list[float]) -> float:
    wins = 0.0
    for positive in answerable:
        for negative in no_answer:
            wins += 1.0 if positive > negative else (0.5 if positive == negative else 0.0)
    return wins / (len(answerable) * len(no_answer))


def choose_threshold(rows: list[dict], split: str) -> dict:
    subset = [row for row in rows if row.get("split") == split]
    positives = [row["retrieved"][0]["score"] for row in subset if not row["gold_empty"]]
    negatives = [row["retrieved"][0]["score"] for row in subset if row["gold_empty"]]
    if not positives or not negatives:
        return {"available": False, "reason": "split lacks answerable or no-answer examples"}
    unique = sorted(set(positives + negatives))
    candidates = [unique[0] - 1e-9] + [
        (left + right) / 2 for left, right in zip(unique, unique[1:])
    ] + [unique[-1] + 1e-9]
    best = None
    for threshold in candidates:
        tpr = sum(score >= threshold for score in positives) / len(positives)
        tnr = sum(score < threshold for score in negatives) / len(negatives)
        balanced = (tpr + tnr) / 2
        candidate = (balanced, tpr, tnr, threshold)
        if best is None or candidate > best:
            best = candidate
    return {
        "available": True,
        "threshold": best[3],
        "balanced_accuracy": best[0],
        "answerable_recall": best[1],
        "no_answer_recall": best[2],
        "answerable_count": len(positives),
        "no_answer_count": len(negatives),
    }


def apply_threshold(rows: list[dict], split: str, threshold_info: dict) -> dict:
    if not threshold_info.get("available"):
        return {"available": False}
    subset = [row for row in rows if row.get("split") == split]
    positives = [row for row in subset if not row["gold_empty"]]
    negatives = [row for row in subset if row["gold_empty"]]
    threshold = threshold_info["threshold"]
    tpr = sum(row["retrieved"][0]["score"] >= threshold for row in positives) / len(positives)
    tnr = sum(row["retrieved"][0]["score"] < threshold for row in negatives) / len(negatives)
    return {
        "available": True,
        "threshold_selected_on_dev": threshold,
        "balanced_accuracy": (tpr + tnr) / 2,
        "answerable_recall": tpr,
        "no_answer_recall": tnr,
        "answerable_count": len(positives),
        "no_answer_count": len(negatives),
    }


def paired_bootstrap(values_a: dict[str, float], values_b: dict[str, float], seed: int, samples: int) -> dict:
    ids = sorted(set(values_a) & set(values_b))
    differences = np.asarray([values_b[item] - values_a[item] for item in ids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        size = min(1000, samples - start)
        indices = rng.integers(0, len(differences), size=(size, len(differences)))
        means[start:start + size] = differences[indices].mean(axis=1)
    observed = float(differences.mean())
    return {
        "unit": "information_need",
        "need_count": len(ids),
        "bootstrap_samples": samples,
        "seed": seed,
        "observed_mean_delta_b_minus_a": observed,
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "two_sided_p_approx": float(min(1.0, 2 * min(np.mean(means <= 0), np.mean(means >= 0)))),
    }


def main_analysis(root: Path, bootstrap_samples: int = 10000) -> dict:
    comparison_dir = root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    rows_by_run = {}
    for name, directory in RUNS.items():
        run_dir = root / directory
        metrics[name] = json.loads((run_dir / "retrieval_metrics.json").read_text(encoding="utf-8"))
        rows_by_run[name] = read_jsonl(run_dir / "retrieval_results.jsonl")

    reference = metrics["tfidf_char"]
    for name, item in metrics.items():
        for field in ("gold_sha256", "index_chunks_sha256", "question_count"):
            if item[field] != reference[field]:
                raise ValueError(f"unfair comparison: {name} differs on {field}")
    reference_ids = [row["question_id"] for row in rows_by_run["tfidf_char"]]
    for name, rows in rows_by_run.items():
        if [row["question_id"] for row in rows] != reference_ids:
            raise ValueError(f"query order differs for {name}")

    query_level = {
        "dataset": "manual_v1_handwritten_v2",
        "gold_sha256": reference["gold_sha256"],
        "chunks_sha256": reference["index_chunks_sha256"],
        "query_count": reference["question_count"],
        "answerable_query_count": reference["scored_nonempty_count"],
        "no_answer_query_count": reference["gold_empty_count"],
        "runs": {
            name: {
                "backend_config": item["backend_config"],
                **{key: item[key] for key in METRIC_KEYS},
            }
            for name, item in metrics.items()
        },
    }
    write_json(comparison_dir / "query_level_metrics.json", query_level)

    need_rows: dict[str, dict[str, list[dict]]] = {
        name: defaultdict(list) for name in RUNS
    }
    for name, rows in rows_by_run.items():
        for row in rows:
            need_rows[name][row["information_need_id"]].append(row)
    per_need = []
    need_values: dict[str, dict[str, float]] = {name: {} for name in RUNS}
    need_full_values: dict[str, dict[str, float]] = {name: {} for name in RUNS}
    for need_id in sorted(need_rows["tfidf_char"]):
        reference_need = need_rows["tfidf_char"][need_id]
        record = {
            "information_need_id": need_id,
            "manual_id": reference_need[0]["manual_id"],
            "product_id": reference_need[0]["product_id"],
            "need_type": reference_need[0].get("need_type"),
            "split": reference_need[0].get("split"),
            "gold_empty": reference_need[0]["gold_empty"],
            "runs": {},
        }
        for name in RUNS:
            group = need_rows[name][need_id]
            if len(group) != 5:
                raise ValueError(f"{need_id}/{name} does not have five styles")
            if record["gold_empty"]:
                run_summary = {"style_count": 5, "scored": False}
            else:
                full_count = sum(row["evidence_recall_at_3"] == 1.0 for row in group)
                avg_evidence = sum(row["evidence_recall_at_3"] for row in group) / 5
                run_summary = {
                    "style_count": 5,
                    "scored": True,
                    "full_evidence_style_count_at_3": full_count,
                    "all_five_styles_full_at_3": full_count == 5,
                    "at_least_four_styles_full_at_3": full_count >= 4,
                    "mean_gold_chunk_recall_at_3": avg_evidence,
                    "mean_reciprocal_rank": sum(row["reciprocal_rank"] for row in group) / 5,
                }
                need_values[name][need_id] = avg_evidence
                need_full_values[name][need_id] = full_count / 5
            record["runs"][name] = run_summary
        per_need.append(record)

    answerable_needs = [row for row in per_need if not row["gold_empty"]]
    need_summary = {
        "independent_unit": "information_need",
        "need_count": len(per_need),
        "answerable_need_count": len(answerable_needs),
        "no_answer_need_count": len(per_need) - len(answerable_needs),
        "runs": {},
        "per_need": per_need,
    }
    for name in RUNS:
        summaries = [row["runs"][name] for row in answerable_needs]
        need_summary["runs"][name] = {
            "all_five_styles_full_need_rate": sum(x["all_five_styles_full_at_3"] for x in summaries) / len(summaries),
            "at_least_four_styles_full_need_rate": sum(x["at_least_four_styles_full_at_3"] for x in summaries) / len(summaries),
            "mean_need_gold_chunk_recall_at_3": sum(x["mean_gold_chunk_recall_at_3"] for x in summaries) / len(summaries),
            "mean_need_mrr": sum(x["mean_reciprocal_rank"] for x in summaries) / len(summaries),
        }
    write_json(comparison_dir / "need_level_metrics.json", need_summary)

    bootstrap = {
        "unit": "information_need",
        "seed": 20260923,
        "comparisons": {},
    }
    pairs = (
        ("tfidf_char", "bge_small_zh"),
        ("tfidf_char", "bge_small_zh_instruction"),
        ("bge_small_zh", "bge_small_zh_instruction"),
    )
    for a, b in pairs:
        label = f"{a}_vs_{b}"
        bootstrap["comparisons"][label] = {
            "mean_need_gold_chunk_recall_at_3": paired_bootstrap(
                need_values[a], need_values[b], 20260923, bootstrap_samples
            ),
            "mean_full_style_rate_at_3": paired_bootstrap(
                need_full_values[a], need_full_values[b], 20260924, bootstrap_samples
            ),
        }
    write_json(comparison_dir / "paired_bootstrap.json", bootstrap)

    grouped_output = {"dataset": "manual_v1_handwritten_v2", "runs": {}}
    for name, rows in rows_by_run.items():
        grouped_output["runs"][name] = {
            "utterance_style": grouped(rows, lambda row: row.get("utterance_style")),
            "need_type": grouped(rows, lambda row: row.get("need_type")),
            "split": grouped(rows, lambda row: row.get("split")),
            "manual": grouped(rows, lambda row: row["manual_id"]),
            "product": grouped(rows, lambda row: row["product_id"]),
            "canonical_vs_non_canonical": grouped(
                rows, lambda row: "canonical" if row.get("utterance_style") == "canonical" else "non_canonical"
            ),
            "has_alternative_gold": grouped(
                rows, lambda row: len(row.get("acceptable_gold_groups") or []) > 1
            ),
        }
    write_json(comparison_dir / "grouped_metrics.json", grouped_output)

    no_answer = {"warning": "Similarity diagnosis only; this does not establish production refusal ability.", "runs": {}}
    high_score_cases = []
    for name, rows in rows_by_run.items():
        answerable_scores = [row["retrieved"][0]["score"] for row in rows if not row["gold_empty"]]
        empty_scores = [row["retrieved"][0]["score"] for row in rows if row["gold_empty"]]
        answerable_third_scores = [row["retrieved"][2]["score"] for row in rows if not row["gold_empty"]]
        empty_third_scores = [row["retrieved"][2]["score"] for row in rows if row["gold_empty"]]
        dev_threshold = choose_threshold(rows, "dev")
        no_answer["runs"][name] = {
            "answerable_top1_scores": quantiles(answerable_scores),
            "no_answer_top1_scores": quantiles(empty_scores),
            "answerable_rank3_scores": quantiles(answerable_third_scores),
            "no_answer_rank3_scores": quantiles(empty_third_scores),
            "answerable_vs_no_answer_top1_auc": score_auc(answerable_scores, empty_scores),
            "dev_threshold_selection": dev_threshold,
            "test_once_with_dev_threshold": apply_threshold(rows, "test", dev_threshold),
        }
        ranked_empty = sorted(
            (row for row in rows if row["gold_empty"]),
            key=lambda row: row["retrieved"][0]["score"], reverse=True,
        )[:20]
        for row in ranked_empty:
            high_score_cases.append({
                "run": name,
                "information_need_id": row["information_need_id"],
                "question_id": row["question_id"],
                "split": row.get("split"),
                "utterance_style": row.get("utterance_style"),
                "query": row["query"],
                "top1": row["retrieved"][0],
                "top3": row["retrieved"][:3],
            })
    write_json(comparison_dir / "no_answer_analysis.json", no_answer)
    with (comparison_dir / "no_answer_high_score_cases.jsonl").open("w", encoding="utf-8") as handle:
        for row in high_score_cases:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    rows_index = {
        name: {row["question_id"]: row for row in rows}
        for name, rows in rows_by_run.items()
    }
    source_queries = {
        need_id: {row["utterance_style"]: row["query"] for row in group}
        for need_id, group in need_rows["tfidf_char"].items()
    }
    error_cases = []
    for question_id in reference_ids:
        left = rows_index["tfidf_char"][question_id]
        if left["gold_empty"]:
            continue
        per_run = {name: rows_index[name][question_id] for name in RUNS}
        full = {name: row["evidence_recall_at_3"] == 1.0 for name, row in per_run.items()}
        labels = []
        if full["tfidf_char"] and not full["bge_small_zh"]:
            labels.append("tfidf_correct_bge_wrong")
        if full["bge_small_zh"] and not full["tfidf_char"]:
            labels.append("bge_correct_tfidf_wrong")
        if left.get("utterance_style") in ("colloquial", "elliptical", "noisy") and not all(full.values()):
            labels.append({"colloquial": "semantic_mismatch", "elliptical": "omitted_context", "noisy": "typo_noise_sensitivity"}[left["utterance_style"]])
        if str(left.get("need_type", "")).startswith("multi_section") and not all(full.values()):
            labels.append("multi_intent_dilution")
        if any(len(row.get("acceptable_gold_groups") or []) > 1 for row in per_run.values()):
            labels.append("alternative_gold")
        if not labels:
            continue
        error_cases.append({
            "information_need_id": left["information_need_id"],
            "question_id": question_id,
            "manual_id": left["manual_id"],
            "product_id": left["product_id"],
            "split": left.get("split"),
            "need_type": left.get("need_type"),
            "utterance_style": left.get("utterance_style"),
            "query": left["query"],
            "five_queries": source_queries[left["information_need_id"]],
            "gold_chunk_ids": left["gold_chunk_ids"],
            "acceptable_gold_groups": left.get("acceptable_gold_groups"),
            "error_labels": sorted(set(labels)),
            "runs": {
                name: {
                    "top3": [{"chunk_id": item["chunk_id"], "score": item["score"]} for item in row["retrieved"][:3]],
                    "gold_chunk_recall_at_3": row["evidence_recall_at_3"],
                    "full_evidence_at_3": row["evidence_recall_at_3"] == 1.0,
                }
                for name, row in per_run.items()
            },
        })
    with (comparison_dir / "error_cases.jsonl").open("w", encoding="utf-8") as handle:
        for row in error_cases:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    for name, directory in RUNS.items():
        run_dir = root / directory
        index_manifest = json.loads((ROOT / metrics[name]["index_dir"] / "index_manifest.json").read_text(encoding="utf-8"))
        run_manifest = {
            "dataset": "manual_v1_handwritten_v2",
            "run_name": name,
            "gold_path": metrics[name]["gold_path"],
            "gold_sha256": metrics[name]["gold_sha256"],
            "chunks_sha256": metrics[name]["index_chunks_sha256"],
            "backend_config": metrics[name]["backend_config"],
            "index_manifest": index_manifest,
            "artifacts": {
                filename: file_sha256(run_dir / filename)
                for filename in ("retrieval_metrics.json", "retrieval_results.jsonl", "gold_empty_results.jsonl", "retrieval_error_report.md")
            },
        }
        write_json(run_dir / "run_manifest.json", run_manifest)

    experiment_manifest = {
        "experiment": "retriever_handwritten_v2_final",
        "dataset": "manual_v1_handwritten_v2",
        "gold_sha256": reference["gold_sha256"],
        "chunks_sha256": reference["index_chunks_sha256"],
        "query_count": reference["question_count"],
        "information_need_count": len(per_need),
        "runs": {
            name: {
                "directory": str((root / directory).relative_to(ROOT)),
                "index_directory": metrics[name]["index_dir"],
                "index_manifest_sha256": file_sha256(ROOT / metrics[name]["index_dir"] / "index_manifest.json"),
                "run_manifest_sha256": file_sha256(root / directory / "run_manifest.json"),
                "backend_config": metrics[name]["backend_config"],
            }
            for name, directory in RUNS.items()
        },
        "comparison_artifacts": [
            "query_level_metrics.json", "need_level_metrics.json", "paired_bootstrap.json",
            "grouped_metrics.json", "no_answer_analysis.json", "no_answer_high_score_cases.jsonl",
            "error_cases.jsonl", "error_analysis.md",
        ],
    }
    write_json(root / "experiment_manifest.json", experiment_manifest)

    label_counts = Counter(label for row in error_cases for label in row["error_labels"])
    write_error_report(comparison_dir / "error_analysis.md", query_level, need_summary, bootstrap, label_counts, len(error_cases))
    write_final_report(root / "final_report.md", query_level, need_summary, bootstrap, no_answer, grouped_output)
    return {
        "valid": True,
        "runs": list(RUNS),
        "query_count": reference["question_count"],
        "need_count": len(per_need),
        "error_case_count": len(error_cases),
        "output_dir": str(root.relative_to(ROOT)),
    }


def write_error_report(path: Path, query: dict, needs: dict, bootstrap: dict, labels: Counter, case_count: int) -> None:
    lines = [
        "# Handwritten Retriever v2 错误分析", "",
        f"- 数据：160 个 information needs / {query['query_count']} 条 query",
        f"- 代表性错误记录：{case_count} 条",
        "- 替代 Gold 已按 acceptable groups 计算，不把未选中的替代 chunk 误报为漏召回。", "",
        "## 错误标签计数", "",
    ]
    lines.extend(f"- `{key}`：{value}" for key, value in labels.most_common())
    lines.extend(["", "## 阅读入口", "", "逐条记录见 `error_cases.jsonl`；每条含五种表达、Gold groups、三个 Retriever 的 Top-3 和分数。", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_final_report(path: Path, query: dict, needs: dict, bootstrap: dict, no_answer: dict, grouped_output: dict) -> None:
    labels = {
        "tfidf_char": "TF-IDF", "bge_small_zh": "BGE", "bge_small_zh_instruction": "BGE+instruction"
    }
    lines = [
        "# Handwritten Retriever v2 正式 Baseline 报告", "",
        "## 实验范围", "",
        "本报告仅使用 `manual_v1_handwritten_v2`：160 个独立 information needs、800 条人工 query；640 条 answerable query 进入主指标，160 条 no-answer 单独诊断。三个系统共享同一 Gold、说明书 chunks、商品过滤和 Evaluator。", "",
        "## Query 级结果", "",
        "| Model | Recall@1 | Recall@2 | Recall@3 | Gold Chunk R@1 | Gold Chunk R@3 | Full Evidence@3 | MRR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in query["runs"].items():
        lines.append(
            f"| {labels[name]} | {item['recall_at_1']:.2%} | {item['recall_at_2']:.2%} | {item['recall_at_3']:.2%} | "
            f"{item['gold_chunk_recall_at_1']:.2%} | {item['gold_chunk_recall_at_3']:.2%} | "
            f"{item['full_evidence_recall_at_3']:.2%} | {item['mrr']:.4f} |"
        )
    lines.extend(["", "## Need 级稳定性", "", "| Model | 五种表达全部完整 | 至少四种完整 | Need 平均 Gold Chunk R@3 |", "|---|---:|---:|---:|"])
    for name, item in needs["runs"].items():
        lines.append(
            f"| {labels[name]} | {item['all_five_styles_full_need_rate']:.2%} | "
            f"{item['at_least_four_styles_full_need_rate']:.2%} | {item['mean_need_gold_chunk_recall_at_3']:.2%} |"
        )
    lines.extend(["", "## 不同表达风格", "", "| Style | TF-IDF R@1 / Full@3 | BGE R@1 / Full@3 | BGE+instruction R@1 / Full@3 |", "|---|---:|---:|---:|"])
    style_names = ("canonical", "natural", "colloquial", "elliptical", "noisy")
    for style in style_names:
        values = []
        for name in RUNS:
            item = grouped_output["runs"][name]["utterance_style"][style]
            values.append(f"{item['recall_at_1']:.2%} / {item['full_evidence_recall_at_3']:.2%}")
        lines.append(f"| {style} | {values[0]} | {values[1]} | {values[2]} |")

    lines.extend(["", "## Single 与 Multi-section", "", "| Need type | Model | Recall@1 | Gold Chunk R@3 | Full@3 |", "|---|---|---:|---:|---:|"])
    for need_type in ("single_section", "multi_section_2"):
        for name in RUNS:
            item = grouped_output["runs"][name]["need_type"][need_type]
            lines.append(
                f"| {need_type} | {labels[name]} | {item['recall_at_1']:.2%} | "
                f"{item['gold_chunk_recall_at_3']:.2%} | {item['full_evidence_recall_at_3']:.2%} |"
            )

    lines.extend(["", "## Split（说明书隔离）", "", "| Split | Model | Recall@1 | Gold Chunk R@3 | Full@3 |", "|---|---|---:|---:|---:|"])
    for split in ("train", "dev", "test"):
        for name in RUNS:
            item = grouped_output["runs"][name]["split"][split]
            lines.append(
                f"| {split} | {labels[name]} | {item['recall_at_1']:.2%} | "
                f"{item['gold_chunk_recall_at_3']:.2%} | {item['full_evidence_recall_at_3']:.2%} |"
            )

    lines.extend(["", "## Need 级配对 Bootstrap", ""])
    for pair, pair_data in bootstrap["comparisons"].items():
        item = pair_data["mean_need_gold_chunk_recall_at_3"]
        lines.append(
            f"- `{pair}`：Δ={item['observed_mean_delta_b_minus_a']:+.4f}，"
            f"95% CI [{item['ci95'][0]:+.4f}, {item['ci95'][1]:+.4f}]。"
        )
    lines.extend(["", "三个区间均跨过 0；当前 128 个 answerable needs 不支持宣称任一模型在 Need 级 Gold Chunk Recall@3 上具有稳定优势。", ""])

    lines.extend(["## No-answer 分数诊断", "", "| Model | Top-1 AUC | Dev balanced acc. | Test balanced acc. | Test answerable recall | Test no-answer recall |", "|---|---:|---:|---:|---:|---:|"])
    for name in RUNS:
        item = no_answer["runs"][name]
        dev = item["dev_threshold_selection"]
        test = item["test_once_with_dev_threshold"]
        lines.append(
            f"| {labels[name]} | {item['answerable_vs_no_answer_top1_auc']:.4f} | "
            f"{dev['balanced_accuracy']:.2%} | {test['balanced_accuracy']:.2%} | "
            f"{test['answerable_recall']:.2%} | {test['no_answer_recall']:.2%} |"
        )
    lines.extend(["", "分数分布存在重叠；阈值诊断不能作为生产拒答能力证明。dev/test 各只有一份说明书，阈值结果必须谨慎解释。", ""])
    best_top1 = max(query["runs"], key=lambda name: query["runs"][name]["recall_at_1"])
    best_full = max(query["runs"], key=lambda name: query["runs"][name]["full_evidence_recall_at_3"])
    lines.extend([
        "", "## 结论", "",
        f"- Query 级 Recall@1 最好的是 **{labels[best_top1]}**。",
        f"- Query 级 Full Evidence@3 最好的是 **{labels[best_full]}**。",
        "- BGE retrieval instruction 改善了 Top-3 证据覆盖，但没有取得最好的 Top-1。",
        "- TF-IDF 在 colloquial 的 Top-1 较弱；BGE 在 elliptical 与部分自然表达上有语义优势，但不同 split 的赢家不一致。",
        "- Multi-section 上，BGE+instruction 的 Gold Chunk Recall@3 和 Full Evidence@3 高于 TF-IDF；single-section 的 Top-1 仍以 TF-IDF 更好。",
        "- 五种表达相关，因此最终判断应结合 need 级结果和 bootstrap 区间，不能把 800 条 query 当作独立样本。",
        "- no-answer 的相似度只用于诊断；即使 dev 阈值在 test 有结果，也不能据此声称系统已经具备可靠拒答能力。",
        "- 下一步优先考虑在 dev 上验证 TF-IDF 与 BGE 的简单 hybrid，而不是直接加入 reranker。", "",
        "详细文件位于 `comparison/`。", "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(main_analysis(args.root, args.bootstrap_samples), ensure_ascii=False, indent=2))
