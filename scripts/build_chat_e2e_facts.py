"""Build an isolated chat fixture from a reviewed sample-scoped fact snapshot.

The source DB is read only. One derived snapshot keeps the existing order,
product, SKU and logistics facts; only a delivery estimate is explicitly
simulated for the multi-turn acceptance scenario.
"""

import hashlib
import json
import os
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/customer_service/customer_service_train.sqlite"
TARGET = ROOT / "data/processed/customer_service/customer_service_chat_e2e.sqlite"
SNAPSHOT_ID = "CHAT-E2E-SNAPSHOT-001"
SAMPLE_ID = "CHAT-E2E-SAMPLE-001"
DELIVERY_ESTIMATE = "预计3至5天，以物流更新为准"


def build(source: Path = SOURCE, target: Path = TARGET) -> dict:
    source, target = source.resolve(), target.resolve()
    if source == target:
        raise ValueError("Fixture target must differ from the source database")
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    descriptor, temporary = tempfile.mkstemp(prefix="chat-e2e-", suffix=".sqlite",
                                             dir=target.parent)
    os.close(descriptor)
    temp_path = Path(temporary)
    try:
        with closing(sqlite3.connect(f"file:{source}?mode=ro", uri=True)) as original:
            with closing(sqlite3.connect(temp_path)) as fixture:
                original.backup(fixture)
                fixture.execute("PRAGMA foreign_keys=ON")
                selected = fixture.execute("""SELECT o.snapshot_id, o.order_id,
                    p.product_id, p.product_name, s.sku_id
                    FROM orders o JOIN products p USING(snapshot_id)
                    JOIN skus s USING(snapshot_id) JOIN logistics l USING(snapshot_id)
                    WHERE o.shipment_status='shipped'
                      AND l.logistics_status='in_transit'
                      AND p.product_id=s.product_id
                      AND p.product_name IS NOT NULL
                    ORDER BY o.snapshot_id LIMIT 1""").fetchone()
                if selected is None:
                    raise ValueError("No existing complete order snapshot is available")
                base_snapshot, order_id, product_id, product_name, sku_id = selected

                columns = [row[1] for row in fixture.execute("PRAGMA table_info(fact_snapshots)")]
                row = dict(zip(columns, fixture.execute(
                    "SELECT * FROM fact_snapshots WHERE snapshot_id=?",
                    (base_snapshot,)).fetchone()))
                fact = json.loads(row["raw_fact_json"])
                fact.update(snapshot_id=SNAPSHOT_ID,
                            estimated_delivery=DELIVERY_ESTIMATE,
                            evidence_source="chat_e2e_curated_from_existing_snapshot")
                row.update(snapshot_id=SNAPSHOT_ID, sample_id=SAMPLE_ID,
                           evidence_source="chat_e2e_curated_from_existing_snapshot",
                           raw_fact_json=json.dumps(fact, ensure_ascii=False))
                fixture.execute(f"INSERT INTO fact_snapshots ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                                tuple(row[name] for name in columns))

                for table in ("orders", "products", "skus", "logistics", "after_sales"):
                    columns = [item[1] for item in fixture.execute(f"PRAGMA table_info({table})")]
                    values = fixture.execute(f"SELECT * FROM {table} WHERE snapshot_id=?",
                                             (base_snapshot,)).fetchone()
                    if values is None:
                        continue
                    record = dict(zip(columns, values))
                    record["snapshot_id"] = SNAPSHOT_ID
                    if table == "logistics":
                        record["estimated_delivery"] = DELIVERY_ESTIMATE
                    fixture.execute(f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                                    tuple(record[name] for name in columns))
                fixture.execute("""CREATE TABLE chat_fixture_metadata (
                    snapshot_id TEXT PRIMARY KEY, source_sha256 TEXT NOT NULL,
                    base_snapshot_id TEXT NOT NULL, simulated_fields_json TEXT NOT NULL)""")
                fixture.execute("INSERT INTO chat_fixture_metadata VALUES (?,?,?,?)",
                                (SNAPSHOT_ID, source_hash, base_snapshot,
                                 json.dumps(["logistics.estimated_delivery"])))
                assert fixture.execute("PRAGMA foreign_key_check").fetchall() == []
                fixture.commit()
        os.replace(temp_path, target)
        return {"target": str(target), "snapshot_id": SNAPSHOT_ID,
                "order_id": order_id, "product_id": product_id,
                "product_name": product_name, "sku_id": sku_id,
                "source_sha256": source_hash}
    finally:
        temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False))
