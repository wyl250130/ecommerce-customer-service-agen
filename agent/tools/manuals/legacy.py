"""Exact-bound category guidance from the original simulated SQLite manuals."""

from __future__ import annotations

import hashlib
import math
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

from ...schemas import ToolResult
from ..common import configure_connection


TOOL = "search_product_manual"


def _error(status: str, code: str) -> ToolResult:
    return ToolResult(False, TOOL, status, error_code=code)


def _sections(document_id: str, content: str) -> list[dict]:
    group, number, rows = "overview", 0, []
    titles = {"overview": "适用范围", "step": "使用步骤", "warning": "注意事项"}
    for raw in content.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == "使用步骤：":
            group, number = "step", 0
            continue
        if line == "注意事项：":
            group, number = "warning", 0
            continue
        number += 1
        section_id = f"{group}-{number}"
        rows.append({"section_id": section_id,
                     "chunk_id": f"{document_id}::{section_id}",
                     "section_title": titles[group], "content": line,
                     "chunk_sha256": hashlib.sha256(line.encode("utf-8")).hexdigest()})
    return rows


def _grams(value: str) -> Counter[str]:
    value = "".join(value.casefold().split())
    return Counter(value[i:i + n] for n in (2, 3)
                   for i in range(max(0, len(value) - n + 1)))


def _rank(chunks: list[dict], query: str, top_k: int) -> list[dict]:
    terms = _grams(query)
    documents = [_grams(row["section_title"] + row["content"]) for row in chunks]
    frequency = Counter(term for doc in documents for term in doc)
    lengths = [sum(doc.values()) for doc in documents]
    average = sum(lengths) / len(lengths) or 1
    scores = []
    for row, doc, length in zip(chunks, documents, lengths):
        score = sum(count * math.log(1 + (len(chunks) - frequency[term] + 0.5) /
                                           (frequency[term] + 0.5)) * doc.get(term, 0) * 2.2 /
                    (doc.get(term, 0) + 1.2 * (0.25 + 0.75 * length / average))
                    for term, count in terms.items() if doc.get(term, 0))
        scores.append(dict(row, score=score))
    return sorted(scores, key=lambda row: (-row["score"], row["chunk_id"]))[:top_k]


class LegacyManualSearchService:
    """Use old approved category manuals, never a cross-snapshot name match."""

    def __init__(self, database_path: Path, top_k: int = 5):
        if not database_path.is_file() or top_k < 1:
            raise ValueError("existing facts SQLite and positive top_k are required")
        self.database_path, self.top_k = database_path, top_k

    def search(self, *, snapshot_id: str, product_id: str | None,
               sku_id: str | None, query: str) -> ToolResult:
        if not isinstance(query, str) or not query.strip():
            return _error("partial", "QUERY_REQUIRED")
        if not product_id:
            return _error("partial", "PRODUCT_NOT_BOUND")
        if not sku_id:
            return _error("partial", "SKU_NOT_BOUND")
        with closing(configure_connection(sqlite3.connect(
                f"file:{self.database_path.resolve()}?mode=ro", uri=True))) as db:
            # The binding must match all three server-controlled identifiers.
            scope = snapshot_id
            rows = self._bound_documents(db, scope, product_id, sku_id)
            if not rows and self._verified_chat_base(db, snapshot_id, product_id, sku_id):
                scope = db.execute("SELECT base_snapshot_id FROM chat_fixture_metadata "
                                   "WHERE snapshot_id=?", (snapshot_id,)).fetchone()[0]
                rows = self._bound_documents(db, scope, product_id, sku_id)
            if not rows:
                return _error("no_match", "MANUAL_BINDING_NOT_FOUND")
            if len(rows) != 1:
                return _error("conflict", "MANUAL_BINDING_AMBIGUOUS")
            document_id, title, content, version = rows[0]
        ranked = [row for row in _rank(_sections(document_id, content), query.strip(), self.top_k)
                  if row["score"] > 0]
        if not ranked:
            return _error("no_match", "MANUAL_SECTION_NOT_FOUND")
        refs = [{"source": "legacy_category_manual", "snapshot_id": snapshot_id,
                 "product_id": product_id, "sku_id": sku_id,
                 "manual_id": document_id, "section_id": item["section_id"],
                 "chunk_sha256": item["chunk_sha256"]} for item in ranked]
        return ToolResult(True, TOOL, "found",
                          data={"snapshot_id": snapshot_id, "product_id": product_id,
                                "sku_id": sku_id, "manual_id": document_id,
                                "manual_title": title, "manual_version": version,
                                "source_tier": "legacy_category",
                                "is_product_specific": False, "query": query.strip(),
                                "results": ranked,
                                "retriever_config": {"retriever_backend": "char_bm25",
                                                     "top_k": self.top_k}},
                          evidence_refs=refs)

    @staticmethod
    def _bound_documents(db, snapshot_id: str, product_id: str, sku_id: str):
        return db.execute("""SELECT d.document_id, d.title, d.content, m.version
            FROM usage_document_bindings AS b
            JOIN usage_documents AS d ON d.document_id=b.document_id
            JOIN usage_document_metadata AS m ON m.document_id=d.document_id
            WHERE b.snapshot_id=? AND b.product_id=? AND b.sku_id=?
              AND m.review_status='approved_for_simulation' AND m.is_simulated=1
            ORDER BY d.document_id""", (snapshot_id, product_id, sku_id)).fetchall()

    @staticmethod
    def _verified_chat_base(db, snapshot_id: str, product_id: str, sku_id: str) -> bool:
        table = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                           "AND name='chat_fixture_metadata'").fetchone()
        if not table:
            return False
        row = db.execute("""SELECT m.base_snapshot_id, p.product_name
            FROM chat_fixture_metadata AS m
            JOIN products AS p ON p.snapshot_id=m.snapshot_id AND p.product_id=?
            JOIN skus AS s ON s.snapshot_id=m.snapshot_id AND s.product_id=? AND s.sku_id=?
            WHERE m.snapshot_id=?""", (product_id, product_id, sku_id, snapshot_id)).fetchone()
        if not row:
            return False
        matching = db.execute("""SELECT 1 FROM usage_document_bindings
            WHERE snapshot_id=? AND product_id=? AND sku_id=? AND product_name=?""",
            (row[0], product_id, sku_id, row[1])).fetchone()
        return bool(matching)
