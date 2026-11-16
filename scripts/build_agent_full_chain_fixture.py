"""Isolated synthetic order for an existing frozen-manual product snapshot."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/customer_service/customer_service_train.sqlite"
TARGET = ROOT / "data/processed/customer_service/customer_service_agent_full_chain.sqlite"
MANUALS = ROOT / "data/processed/product_manuals/frozen_v1/product_manuals_frozen_v1.jsonl"
SNAPSHOT_ID = "GOLD-SNAP-SEMANTIC-GOLD-TRAIN-000551"
PRODUCT_ID = "BABY-COTTON-SWAB-551"
SKU_ID = "FINE-TIP"
MANUAL_ID = "MANUAL-BABY-COTTON-SWAB-551-V1"
ORDER_ID = "E2E-MANUAL-551-001"


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(target: Path = TARGET) -> dict:
    source = SOURCE.resolve()
    target = target.resolve()
    if source == target:
        raise ValueError("target must differ from the source database")
    source_hash = hash_file(source)
    with MANUALS.open(encoding="utf-8") as source_file:
        manuals = [json.loads(line) for line in source_file if line.strip()]
    selected = next(row for row in manuals if row["manual_id"] == MANUAL_ID)
    if selected["product_binding"]["snapshot_id"] != SNAPSHOT_ID or (
        selected["product_binding"]["product_id"], selected["product_binding"]["sku_id"]
    ) != (PRODUCT_ID, SKU_ID):
        raise ValueError("frozen manual binding changed")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="agent-full-chain-", suffix=".sqlite", dir=target.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        with closing(sqlite3.connect(f"file:{source}?mode=ro", uri=True)) as original:
            with closing(sqlite3.connect(temporary_path)) as fixture:
                original.backup(fixture)
                fixture.execute("PRAGMA foreign_keys=ON")
                product = fixture.execute("SELECT product_id, product_name FROM products WHERE snapshot_id=?",
                                          (SNAPSHOT_ID,)).fetchone()
                sku = fixture.execute("SELECT product_id, sku_id FROM skus WHERE snapshot_id=?",
                                      (SNAPSHOT_ID,)).fetchone()
                if product != (PRODUCT_ID, selected["product_binding"]["product_name"]) or sku != (PRODUCT_ID, SKU_ID):
                    raise ValueError("source product/SKU no longer matches frozen manual")
                if fixture.execute("SELECT 1 FROM orders WHERE snapshot_id=?", (SNAPSHOT_ID,)).fetchone():
                    raise ValueError("source snapshot already has an order")
                fixture.execute("""INSERT INTO orders
                    (snapshot_id, order_id, order_record_status, order_item_status,
                     payment_status, shipment_status, extra_json)
                    VALUES (?, ?, 'matched_context_record', 'linked_product',
                            'paid', 'not_shipped', '{}')""", (SNAPSHOT_ID, ORDER_ID))
                fixture.execute("""CREATE TABLE agent_full_chain_fixture_metadata (
                    snapshot_id TEXT PRIMARY KEY, order_id TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL, frozen_manual_sha256 TEXT NOT NULL,
                    synthetic_fields_json TEXT NOT NULL)""")
                fixture.execute("INSERT INTO agent_full_chain_fixture_metadata VALUES (?,?,?,?,?)",
                                (SNAPSHOT_ID, ORDER_ID, source_hash, hash_file(MANUALS),
                                 json.dumps(["orders.*"])))
                if fixture.execute("PRAGMA foreign_key_check").fetchall():
                    raise ValueError("fixture foreign key violation")
                fixture.commit()
        os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)
    if hash_file(source) != source_hash:
        raise RuntimeError("source database changed while building fixture")
    manifest = {"path": str(target), "snapshot_id": SNAPSHOT_ID,
            "order_id": ORDER_ID, "product_id": PRODUCT_ID, "sku_id": SKU_ID,
            "manual_id": MANUAL_ID, "source_sha256": source_hash,
            "frozen_manual_sha256": hash_file(MANUALS),
            "synthetic_fields": ["orders.*"],
            "question": f"查询订单 {ORDER_ID}：发货了吗？购买的商品如何安全用于耳部清洁？",
            "reference_response": (f"模拟订单 {ORDER_ID} 已付款、未发货，关联商品是婴儿细轴棉签。"
                                   "该商品的模拟说明书只允许轻柔清洁看得见的外耳廓部位，"
                                   "不得将棉签伸入耳道。")}
    target.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False,
                                                   indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
