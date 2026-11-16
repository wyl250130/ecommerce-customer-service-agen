import sqlite3
import unittest
from contextlib import closing
from pathlib import Path

from agent.tools.orders.info import get_order_info
from agent.tools.common import configure_connection


DB = (Path(__file__).resolve().parents[2] /
      "data/processed/customer_service/customer_service_train.sqlite")


class OrderProductDetailTest(unittest.TestCase):
    def test_order_detail_uses_only_its_snapshot(self):
        with closing(configure_connection(sqlite3.connect(DB))) as db:
            with_product = db.execute("""SELECT o.snapshot_id, o.order_id, p.product_id,
                p.product_name FROM orders o JOIN products p USING(snapshot_id)
                WHERE p.product_id IS NOT NULL AND p.product_name IS NOT NULL LIMIT 1""").fetchone()
            without_product = db.execute("""SELECT o.snapshot_id, o.order_id FROM orders o
                LEFT JOIN products p USING(snapshot_id) WHERE p.snapshot_id IS NULL LIMIT 1""").fetchone()
            self.assertIsNotNone(with_product)
            self.assertIsNotNone(without_product)

            sid, order_id, product_id, product_name = with_product
            found = get_order_info(db, sid, order_id)
            self.assertTrue(found.ok)
            self.assertEqual(found.data["product_id"], product_id)
            self.assertEqual(found.data["product_name"], product_name)
            self.assertTrue(found.data["product_info_available"])
            self.assertIn("products", [r["source"] for r in found.evidence_refs])

            sid, order_id = without_product
            missing = get_order_info(db, sid, order_id)
            self.assertTrue(missing.ok)
            self.assertIsNone(missing.data["product_name"])
            self.assertFalse(missing.data["product_info_available"])
            self.assertEqual([r["source"] for r in missing.evidence_refs], ["orders"])

            self.assertEqual(get_order_info(db, sid, with_product[1]).status, "no_match")
