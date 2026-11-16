"""Build same-snapshot order/product/manual consistency scenarios for Agent tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

from rag.unified_manuals import DEFAULT_CORPUS, UnifiedManualCorpus


ROOT = Path(__file__).resolve().parents[1]
TRAIN_DB = ROOT / "data/processed/customer_service/customer_service_train.sqlite"
DEMO_DB = ROOT / "data/processed/customer_service/customer_service_chat_e2e.sqlite"
DEFAULT_OUTPUT = ROOT / "data/processed/agent_e2e_v1"
STATUS_TEXT = {"not_shipped": "未发货", "delivered": "已送达", "shipped": "已发货"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_db(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def build(output: Path = DEFAULT_OUTPUT, corpus_dir: Path = DEFAULT_CORPUS) -> dict:
    corpus = UnifiedManualCorpus(corpus_dir)
    cases = []
    with closing(read_db(TRAIN_DB)) as db:
        rows = db.execute("""
            SELECT o.snapshot_id, o.order_id, o.shipment_status,
                   p.product_id, p.product_name, s.sku_id, b.document_id AS manual_id
            FROM orders AS o
            JOIN products AS p ON p.snapshot_id = o.snapshot_id
            JOIN skus AS s ON s.snapshot_id = o.snapshot_id AND s.product_id = p.product_id
            JOIN usage_document_bindings AS b ON b.snapshot_id = o.snapshot_id
                AND b.product_id = p.product_id AND b.sku_id = s.sku_id
            WHERE o.snapshot_id LIKE '%TRAIN%' AND o.shipment_status IN ('not_shipped', 'delivered')
            ORDER BY o.snapshot_id, b.document_id
        """).fetchall()
        for row in rows:
            cases.append(_case(dict(row), "train", corpus))

    with closing(read_db(DEMO_DB)) as db:
        row = db.execute("""
            SELECT o.snapshot_id, o.order_id, o.shipment_status,
                   p.product_id, p.product_name, s.sku_id
            FROM orders o JOIN products p USING(snapshot_id)
            JOIN skus s ON s.snapshot_id = o.snapshot_id AND s.product_id = p.product_id
            WHERE o.snapshot_id = 'CHAT-E2E-SNAPSHOT-001'
        """).fetchone()
        if row is None:
            raise ValueError("chat demo fixture is missing")
        cases.append(_case(dict(row), "demo", corpus))

    if len({case["scenario_id"] for case in cases}) != len(cases):
        raise ValueError("duplicate scenario IDs")
    output.mkdir(parents=True, exist_ok=True)
    scenarios_path = output / "scenarios.jsonl"
    with scenarios_path.open("w", encoding="utf-8") as stream:
        for case in cases:
            stream.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "version": "agent_e2e_same_snapshot_v1",
        "counts": {"scenarios": len(cases), "train": sum(c["split"] == "train" for c in cases),
                   "demo": sum(c["split"] == "demo" for c in cases),
                   "distinct_legacy_manuals": len({c["manual_id"] for c in cases})},
        "by_manual": dict(sorted(Counter(c["manual_id"] for c in cases).items())),
        "source_sha256": {"train_sqlite": sha256(TRAIN_DB), "demo_sqlite": sha256(DEMO_DB),
                          "unified_bindings": sha256(corpus_dir / "bindings.jsonl"),
                          "unified_chunks": sha256(corpus_dir / "chunks.jsonl")},
        "scenarios_sha256": sha256(scenarios_path),
        "limitations": ["synthetic facts, not real commerce events",
                        "category-level manuals, not SKU-specific manufacturer manuals",
                        "reference responses are generated factual checks, not model outputs or human Gold"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False,
                                                    indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _case(row: dict, split: str, corpus: UnifiedManualCorpus) -> dict:
    if not all(row.get(field) for field in ("snapshot_id", "order_id", "product_id", "product_name", "sku_id")):
        raise ValueError("scenario has incomplete order/product/SKU identity")
    candidates = corpus.candidates(row["snapshot_id"], row["product_id"], row["sku_id"])
    if not candidates or {item["source_tier"] for item in candidates} != {"legacy_category"}:
        raise ValueError(f"missing exact legacy binding: {row['snapshot_id']}")
    manual_ids = {item["manual_id"] for item in candidates}
    if "manual_id" in row and row["manual_id"] not in manual_ids:
        raise ValueError(f"source binding disagrees with corpus: {row['snapshot_id']}")
    manual_id = row.get("manual_id") or next(iter(manual_ids))
    steps = [item for item in candidates if item["manual_id"] == manual_id
             and item["section_id"] == "step-1"]
    if len(steps) != 1:
        raise ValueError(f"missing unique step-1: {manual_id}")
    step = steps[0]
    status = STATUS_TEXT.get(row["shipment_status"])
    if status is None:
        raise ValueError(f"ambiguous order status: {row['shipment_status']}")
    title = next(item["title"] for item in json_lines(corpus.directory / "manuals.jsonl")
                 if item["manual_id"] == manual_id)
    response = (f"模拟订单 {row['order_id']} 当前记录为{status}，关联商品是{row['product_name']}"
                f"（SKU：{row['sku_id']}）。可用的类别级模拟说明《{title}》中有一条使用步骤："
                f"{step['content']} 这不是该 SKU 的厂商专属说明；具体参数请以商品包装为准。")
    return {
        "scenario_id": f"E2E-{split.upper()}-{row['snapshot_id']}", "split": split,
        "snapshot_id": row["snapshot_id"], "order_id": row["order_id"],
        "shipment_status": row["shipment_status"], "product_id": row["product_id"],
        "product_name": row["product_name"], "sku_id": row["sku_id"],
        "manual_id": manual_id, "manual_source_tier": "legacy_category",
        "manual_chunk_id": step["chunk_id"], "manual_chunk_sha256": step["content_sha256"],
        "manual_chunk_content": step["content"],
        "user_query": f"查询订单 {row['order_id']}，告诉我发货状态、对应商品，以及说明书中的一条使用步骤。",
        "reference_response": response,
        "reference_kind": "generated_fact_consistency_reference",
        "source_database": "chat_demo" if split == "demo" else "train_sqlite",
    }


def json_lines(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.corpus)["counts"], ensure_ascii=False))
