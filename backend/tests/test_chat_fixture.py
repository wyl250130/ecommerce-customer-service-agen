import hashlib
import json
import sqlite3
from contextlib import closing

from agent.tools.common import configure_connection
from agent.tools.orders.info import get_order_info
from agent.tools.shipping.logistics import get_logistics
from scripts.build_chat_e2e_facts import SOURCE, SNAPSHOT_ID, build


def test_curated_chat_fixture_preserves_source_and_links_facts(tmp_path):
    before = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    target = tmp_path / "chat.sqlite"
    manifest = build(target=target)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == before
    assert manifest["source_sha256"] == before

    with closing(configure_connection(sqlite3.connect(target))) as db:
        order = get_order_info(db, SNAPSHOT_ID, manifest["order_id"])
        logistics = get_logistics(db, SNAPSHOT_ID, manifest["order_id"])
        assert order.ok and order.data["product_info_available"]
        assert order.data["product_id"] == manifest["product_id"]
        assert order.data["product_name"] == manifest["product_name"]
        assert order.data["sku_id"] == manifest["sku_id"]
        assert logistics.ok and logistics.data["estimated_delivery"]
        metadata = db.execute("SELECT simulated_fields_json FROM chat_fixture_metadata").fetchone()[0]
        assert json.loads(metadata) == ["logistics.estimated_delivery"]
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
        assert get_order_info(db, "wrong-snapshot", manifest["order_id"]).status == "no_match"
