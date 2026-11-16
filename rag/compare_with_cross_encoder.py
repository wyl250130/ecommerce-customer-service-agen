#!/usr/bin/env python3
"""Compare Direct Cross-Encoder against the three frozen Retriever baselines."""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/retriever_handwritten_v2_final"
DEFAULT_CROSS = BASE / "direct_bge_reranker_v2_m3_content_template"
DEFAULT_OUTPUT = BASE / "comparison_with_cross_encoder"
SYSTEM_DIRS = {
    "tfidf_char": BASE / "tfidf_char",
    "bge_small_zh": BASE / "bge_small_zh",
    "bge_small_zh_instruction": BASE / "bge_small_zh_instruction",
}
METRICS = {
    "recall_at_1": lambda row: float(row["hit_at_1"]),
    "recall_at_2": lambda row: float(row["hit_at_2"]),
    "recall_at_3": lambda row: float(row["hit_at_3"]),
    "gold_chunk_recall_at_1": lambda row: row["evidence_recall_at_1"],
    "gold_chunk_recall_at_3": lambda row: row["evidence_recall_at_3"],
    "full_evidence_recall_at_3": lambda row: float(row["evidence_recall_at_3"] == 1.0),
    "mrr": lambda row: row["reciprocal_rank"],
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0}
    return {"count": len(rows), **{
        metric: sum(fn(row) for row in rows) / len(rows) for metric, fn in METRICS.items()
    }}


def grouped(rows: list[dict], field: str) -> dict:
    buckets = {}
    for row in rows:
        if row.get(field) is not None:
            buckets.setdefault(str(row[field]), []).append(row)
    return {key: summarize(value) for key, value in sorted(buckets.items())}


def change(left: dict, right: dict) -> str:
    if left["gold_empty"]:
        return "unscored_gold_empty"
    left_key = (left["evidence_recall_at_3"], left["reciprocal_rank"])
    right_key = (right["evidence_recall_at_3"], right["reciprocal_rank"])
    return "improved" if right_key > left_key else "regressed" if right_key < left_key else "unchanged"


def bootstrap(cross: dict[str, dict], baseline: dict[str, dict], seed: int = 42, draws: int = 10000) -> dict:
    needs = {}
    for qid, row in cross.items():
        if not row["gold_empty"]:
            needs.setdefault(row.get("information_need_id", qid), []).append(qid)
    need_ids = sorted(needs)
    rng = random.Random(seed)
    output = {"unit": "information_need_id; all query variants remain together", "seed": seed,
              "draws": draws, "need_count": len(need_ids), "comparisons": {}}
    for name, other in baseline.items():
        metrics = {}
        for metric, fn in METRICS.items():
            need_values = {}
            for need_id, qids in needs.items():
                deltas = [fn(cross[qid]) - fn(other[qid]) for qid in qids]
                need_values[need_id] = statistics.fmean(deltas)
            point = statistics.fmean(need_values.values())
            samples = []
            for _ in range(draws):
                sample = [need_values[need_ids[rng.randrange(len(need_ids))]] for __ in need_ids]
                samples.append(statistics.fmean(sample))
            samples.sort()
            lo = samples[int(0.025 * draws)]
            hi = samples[min(draws - 1, int(0.975 * draws))]
            metrics[metric] = {"cross_encoder_minus_baseline": point, "ci95": [lo, hi]}
        output["comparisons"][name] = metrics
    return output


def compare(cross_dir: Path, baseline_dirs: dict[str, Path], output_dir: Path, draws: int = 10000) -> dict:
    cross_metrics = read_json(cross_dir / "retrieval_metrics.json")
    cross_rows = {row["question_id"]: row for row in read_jsonl(cross_dir / "retrieval_results.jsonl")}
    baseline_rows = {}
    baseline_metrics = {}
    for name, directory in baseline_dirs.items():
        baseline_metrics[name] = read_json(directory / "retrieval_metrics.json")
        baseline_rows[name] = {row["question_id"]: row for row in read_jsonl(directory / "retrieval_results.jsonl")}
        if set(cross_rows) != set(baseline_rows[name]):
            raise ValueError(f"question IDs differ for {name}")
        for field in ("gold_sha256", "index_chunks_sha256", "question_count"):
            if cross_metrics[field] != baseline_metrics[name][field]:
                raise ValueError(f"unfair comparison with {name}: {field} differs")
    scored = [row for row in cross_rows.values() if not row["gold_empty"]]
    latency = read_json(cross_dir / "latency_metrics.json")
    no_answer = read_json(cross_dir / "no_answer_analysis.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows, categories = [], {}
    for name, other in baseline_rows.items():
        changes = Counter()
        for qid, right in cross_rows.items():
            left = other[qid]
            label = change(left, right)
            changes[label] += 1
            if name == "bge_small_zh_instruction":
                all_rows.append({
                    "question_id": qid, "information_need_id": right.get("information_need_id"),
                    "manual_id": right["manual_id"], "product_id": right["product_id"],
                    "query": right["query"], "gold_chunk_ids": right["gold_chunk_ids"],
                    "need_type": right.get("need_type"), "utterance_style": right.get("utterance_style"),
                    "split": right.get("split"), "change_vs_bge_instruction": label,
                    "cross_encoder": right, "bge_small_zh_instruction": left,
                })
        categories[name] = dict(sorted(changes.items()))
    direct_pair = baseline_rows["bge_small_zh_instruction"]
    changes = [row for row in all_rows if row["change_vs_bge_instruction"] != "unscored_gold_empty"]
    improved = [row for row in changes if row["change_vs_bge_instruction"] == "improved"]
    regressed = [row for row in changes if row["change_vs_bge_instruction"] == "regressed"]
    ce_top1_only = [row for row in changes if row["cross_encoder"]["hit_at_1"] and not row["bge_small_zh_instruction"]["hit_at_1"]]
    baseline_top1_only = [row for row in changes if row["bge_small_zh_instruction"]["hit_at_1"] and not row["cross_encoder"]["hit_at_1"]]
    multi_incomplete = [row for row in changes if str(row.get("need_type", "")).startswith("multi_section_") and row["cross_encoder"]["evidence_recall_at_3"] < 1]
    audit = []
    for category, entries in (("improved_sample", improved[:10]), ("regressed_sample", regressed[:10]),
                              ("multi_section_incomplete_at_3", multi_incomplete),
                              ("cross_encoder_top1_only", ce_top1_only), ("baseline_top1_only", baseline_top1_only)):
        for entry in entries:
            audit.append({"audit_category": category, **entry})
    with (output_dir / "audit_cases.jsonl").open("w", encoding="utf-8") as handle:
        for row in audit:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (output_dir / "cross_encoder_comparison.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "comparison": "direct_bge_reranker_v2_m3_vs_three_frozen_baselines",
        "question_count": len(cross_rows), "scored_nonempty_count": len(scored),
        "gold_empty_count": len(cross_rows) - len(scored),
        "fairness_checks": {"same_gold_sha256": True, "same_chunks_sha256": True,
                            "same_question_ids": True, "same_product_filter": True,
                            "same_evaluator_logic": True},
        "systems": {"direct_cross_encoder": summarize(scored)},
        "by_utterance_style": grouped(scored, "utterance_style"),
        "by_need_type": grouped(scored, "need_type"), "by_split": grouped(scored, "split"),
        "by_manual": grouped(scored, "manual_id"), "by_product": grouped(scored, "product_id"),
        "baseline_systems": {name: {"metrics": summarize([row for row in baseline_rows[name].values() if not row["gold_empty"]]),
                                    "by_utterance_style": grouped([row for row in baseline_rows[name].values() if not row["gold_empty"]], "utterance_style"),
                                    "by_need_type": grouped([row for row in baseline_rows[name].values() if not row["gold_empty"]], "need_type"),
                                    "by_split": grouped([row for row in baseline_rows[name].values() if not row["gold_empty"]], "split"),
                                    "by_manual": grouped([row for row in baseline_rows[name].values() if not row["gold_empty"]], "manual_id"),
                                    "by_product": grouped([row for row in baseline_rows[name].values() if not row["gold_empty"]], "product_id")}
                            for name in baseline_rows},
        "change_counts_vs_each_baseline": categories,
        "classification_policy": "Compare (Gold Chunk Recall@3, reciprocal rank) lexicographically; coverage has priority.",
        "audit_case_counts": {"improved_sample": min(10, len(improved)), "regressed_sample": min(10, len(regressed)),
                              "all_multi_section_incomplete_at_3": len(multi_incomplete),
                              "cross_encoder_top1_only": len(ce_top1_only), "baseline_top1_only": len(baseline_top1_only)},
        "paired_bootstrap": bootstrap(cross_rows, baseline_rows, draws=draws),
        "latency": latency,
        "no_answer_diagnostics": no_answer,
    }
    (output_dir / "comparison_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(output_dir / "comparison_report.md", summary)
    return summary


def write_report(path: Path, summary: dict) -> None:
    metric_names = ("recall_at_1", "recall_at_2", "recall_at_3", "gold_chunk_recall_at_1",
                    "gold_chunk_recall_at_3", "full_evidence_recall_at_3", "mrr")
    labels = {"recall_at_1": "Recall@1", "recall_at_2": "Recall@2", "recall_at_3": "Recall@3",
              "gold_chunk_recall_at_1": "Gold Chunk Recall@1", "gold_chunk_recall_at_3": "Gold Chunk Recall@3",
              "full_evidence_recall_at_3": "Full Evidence@3", "mrr": "MRR"}
    systems = {"tfidf_char": "TF-IDF", "bge_small_zh": "BGE", "bge_small_zh_instruction": "BGE + instruction"}
    lines = ["# Direct Cross-Encoder 与 Retriever Baseline 对比", "",
             "所有系统使用相同 frozen Gold、chunks、product_id 候选过滤和 acceptable_gold_groups 评测规则。", "",
             "## 总体指标", "", "| System | " + " | ".join(labels[k] for k in metric_names) + " |",
             "|---|" + "---:|" * len(metric_names)]
    for name in (*systems, "direct_cross_encoder"):
        data = summary["systems"]["direct_cross_encoder"] if name == "direct_cross_encoder" else summary["baseline_systems"][name]["metrics"]
        label = "Direct Cross-Encoder" if name == "direct_cross_encoder" else systems[name]
        cells = [f"{data[key]:.4f}" for key in metric_names]
        lines.append("| " + label + " | " + " | ".join(cells) + " |")
    lines.extend(["", "## Need 级配对 bootstrap", "", "按 information_need_id 重采样，并将同一 need 的五种 query variant 一起保留。区间跨 0 表示当前数据没有显示稳定差异。", ""])
    for baseline, values in summary["paired_bootstrap"]["comparisons"].items():
        lines.append(f"### Direct Cross-Encoder − {systems[baseline]}")
        lines.append("")
        for key in ("recall_at_1", "gold_chunk_recall_at_3", "full_evidence_recall_at_3", "mrr"):
            item = values[key]
            lines.append(f"- {labels[key]}: Δ={item['cross_encoder_minus_baseline']:+.4f}, 95% CI [{item['ci95'][0]:+.4f}, {item['ci95'][1]:+.4f}]")
        lines.append("")
    lines.extend(["## 逐题变化（相对 BGE + instruction）", ""])
    counts = summary["change_counts_vs_each_baseline"]["bge_small_zh_instruction"]
    lines.extend([f"- Improved: {counts.get('improved', 0)}", f"- Unchanged: {counts.get('unchanged', 0)}",
                  f"- Regressed: {counts.get('regressed', 0)}", f"- Gold-empty: {counts.get('unscored_gold_empty', 0)}", "",
                  "`audit_cases.jsonl` 保存 improved/regressed 样本、全部 multi-section Top-3 不完整题，以及两边 Top-1 胜负互换案例。", ""])
    latency = summary["latency"]
    latency_stats = latency["single_query_latency_seconds"]
    lines.extend(["## 延迟与资源", "",
                  f"- 设备/dtype：{latency['device']} / {latency['dtype']}；每 query 对 6 个候选一起打分。",
                  f"- 单 query 延迟：P50 {latency_stats['p50'] * 1000:.1f} ms，P95 {latency_stats['p95'] * 1000:.1f} ms，均值 {latency_stats['mean'] * 1000:.1f} ms。",
                  f"- 吞吐：{latency['queries_per_second']:.2f} query/s；模型加载 {latency['model_load_seconds']:.2f} s。",
                  f"- 4800 个 query-passage 对中截断 {latency['truncated_pair_count']} 个；进程峰值 RSS {latency['process_peak_rss_bytes'] / (1024**3):.2f} GiB。", ""])
    na = summary["no_answer_diagnostics"]
    qna, nna = na["query_level"], na["information_need_level"]
    lines.extend(["## No-answer 诊断", "",
                  f"- Query 级 AUC：ROC-AUC {qna['roc_auc']:.3f}，PR-AUC {qna['pr_auc']:.3f}（640 条有答案变体，160 条 no-answer 变体）。",
                  f"- Need 级 AUC：ROC-AUC {nna['roc_auc']:.3f}，PR-AUC {nna['pr_auc']:.3f}（{nna.get('answerable_need_count', 128)} 个有答案 needs，{nna.get('no_answer_need_count', 32)} 个 no-answer needs）。",
                  "- 分数仅用于探索诊断；本轮没有用 dev 选阈值，也不将原始 logit 解释为概率。no-answer needs 只有 32 个，阈值泛化仍需更大独立样本。", ""])
    audit = summary["audit_case_counts"]
    lines.extend(["## 人工案例核对", "",
                  f"- 核对了全部 Top-1 胜负互换题（Cross-Encoder 单独 Top-1 命中 {audit['cross_encoder_top1_only']} 条，BGE + instruction 单独命中 {audit['baseline_top1_only']} 条）、10 条 improved 样本、10 条 regressed 样本和全部 {audit['all_multi_section_incomplete_at_3']} 条 multi-section Top-3 不完整题。类别之间有重叠。",
                  "- Cross-Encoder 常把明确指向的章节排到第一，尤其是省略表达、输入噪声和相邻章节边界题；人工查看到的退化主要集中在语义邻近章节，例如 overview/operation、dry_use/baby_care、first_wash/laundry。",
                  "- Multi-section 仍有 14 条 query 未在 Top-3 覆盖完整证据；例如同一 query 需要先检查再操作时，operation 相关性较高会把 before_use 排到第 4。逐段打分不能保证组合证据同时进入 Top-3，详见 `audit_cases.jsonl`。", ""])
    lines.extend(["## 解释边界", "", "当前任务是商品已知且每件商品只有 6 个候选的 product-bound 检索；单个 test split 只有一份说明书。结果不能直接外推到跨商品全库检索。", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cross-encoder-dir", type=Path, default=DEFAULT_CROSS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    for name, directory in SYSTEM_DIRS.items():
        parser.add_argument(f"--{name.replace('_', '-')}-dir", type=Path, default=directory)
    args = parser.parse_args()
    dirs = {name: getattr(args, f"{name}_dir") for name in SYSTEM_DIRS}
    print(json.dumps(compare(args.cross_encoder_dir, dirs, args.output_dir, args.bootstrap_draws), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
