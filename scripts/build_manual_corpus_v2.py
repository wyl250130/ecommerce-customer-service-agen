#!/usr/bin/env python3
"""Export old category manuals and frozen product manuals without changing either source."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data/processed/customer_service/customer_service_train.sqlite"
CHAT_DB = ROOT / "data/processed/customer_service/customer_service_chat_e2e.sqlite"
FROZEN = ROOT / "data/processed/product_manuals/frozen_v1"
GOLD = ROOT / "data/processed/retriever_eval/manual_v1_handwritten_v2_compiled.jsonl"
OUTPUT = ROOT / "data/processed/product_manuals/unified_v2"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def connect_readonly(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(path)
    db = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def key(snapshot_id: str, product_id: str, sku_id: str) -> str:
    return json.dumps([snapshot_id, product_id, sku_id], ensure_ascii=False, separators=(",", ":"))


def old_sections(content: str) -> list[tuple[str, str, str]]:
    sections = []
    group = "overview"
    index = 0
    for raw in content.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == "使用步骤：":
            group, index = "step", 0
            continue
        if line == "注意事项：":
            group, index = "warning", 0
            continue
        index += 1
        title = {"overview": "适用范围", "step": "使用步骤", "warning": "注意事项"}[group]
        sections.append((f"{group}-{index}", title, line))
    return sections


def build(db_path: Path = DEFAULT_DB, chat_db_path: Path = CHAT_DB,
          output_dir: Path = OUTPUT) -> dict:
    frozen_manual_path = FROZEN / "product_manuals_frozen_v1.jsonl"
    frozen_chunk_path = FROZEN / "product_manual_chunks_frozen_v1.jsonl"
    frozen = read_jsonl(frozen_manual_path)
    frozen_chunks = read_jsonl(frozen_chunk_path)
    frozen_by_id = {row["manual_id"]: row for row in frozen}
    if len(frozen_by_id) != len(frozen) or len(frozen_chunks) != 48:
        raise ValueError("unexpected frozen manual corpus; validate frozen_v1 first")

    with closing(connect_readonly(db_path)) as db:
        old_docs = [dict(row) for row in db.execute(
            "SELECT document_id, title, content, source FROM usage_documents ORDER BY document_id")]
        old_bindings = [dict(row) for row in db.execute(
            "SELECT document_id, snapshot_id, product_id, sku_id, product_name, match_basis "
            "FROM usage_document_bindings ORDER BY snapshot_id, document_id")]
        products = {row["snapshot_id"]: dict(row) for row in db.execute(
            "SELECT snapshot_id, product_id, product_name, material, spec, size "
            "FROM products ORDER BY snapshot_id")}
        skus = {row["snapshot_id"]: dict(row) for row in db.execute(
            "SELECT snapshot_id, product_id, sku_id FROM skus ORDER BY snapshot_id")}
    old_ids = {row["document_id"] for row in old_docs}
    if len(old_ids) != len(old_docs) or old_ids & set(frozen_by_id):
        raise ValueError("duplicate or colliding manual IDs")

    manuals = [{"manual_id": row["document_id"], "title": row["title"],
                "source_tier": "legacy_category", "source": row["source"],
                "scope": "category_guidance", "original_content_sha256": hashlib.sha256(
                    row["content"].encode("utf-8")).hexdigest()} for row in old_docs]
    manuals.extend({"manual_id": row["manual_id"],
                    "title": row["product_binding"]["product_name"],
                    "source_tier": "frozen_product", "source": row["source"],
                    "scope": "exact_product_sku",
                    "original_content_sha256": row["content_sha256"]} for row in frozen)

    chunks = []
    for row in old_docs:
        for section_id, title, content in old_sections(row["content"]):
            chunks.append({"chunk_id": f"{row['document_id']}::{section_id}",
                           "manual_id": row["document_id"], "section_id": section_id,
                           "section_title": title, "content": content,
                           "source_tier": "legacy_category",
                           "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()})
    for row in frozen_chunks:
        if row["manual_id"] not in frozen_by_id:
            raise ValueError("orphan frozen chunk")
        chunks.append({"chunk_id": row["chunk_id"], "manual_id": row["manual_id"],
                       "section_id": row["section_id"], "section_title": row["section_title"],
                       "content": row["content"], "source_tier": "frozen_product",
                       "content_sha256": row["chunk_sha256"]})
    if len({row["chunk_id"] for row in chunks}) != len(chunks):
        raise ValueError("duplicate chunk IDs")

    bindings = []
    for row in old_bindings:
        product, sku = products.get(row["snapshot_id"]), skus.get(row["snapshot_id"])
        if row["document_id"] not in old_ids or not product or not sku or (
            product["product_id"] != row["product_id"] or
            sku["product_id"] != row["product_id"] or
            sku["sku_id"] != row["sku_id"]
        ):
            raise ValueError(f"invalid legacy binding: {row['snapshot_id']}")
        bindings.append({"binding_key": key(row["snapshot_id"], row["product_id"], row["sku_id"]),
                         "snapshot_id": row["snapshot_id"], "product_id": row["product_id"],
                         "sku_id": row["sku_id"], "product_name": row["product_name"],
                         "manual_id": row["document_id"], "source_tier": "legacy_category",
                         "match_basis": row["match_basis"]})
    for row in frozen:
        binding = row["product_binding"]
        # Frozen source snapshots may be in dev/test and absent from the train SQLite.
        bindings.append({"binding_key": key(binding["snapshot_id"], binding["product_id"], binding["sku_id"]),
                         "snapshot_id": binding["snapshot_id"], "product_id": binding["product_id"],
                         "sku_id": binding["sku_id"], "product_name": binding["product_name"],
                         "manual_id": row["manual_id"], "source_tier": "frozen_product",
                         "match_basis": "frozen_exact_product_sku"})

    # Only inherit a category binding for the explicit chat fixture when its
    # recorded base snapshot and product/SKU facts agree exactly.
    derived = []
    if chat_db_path.is_file():
        with closing(connect_readonly(chat_db_path)) as chat:
            for fixture in chat.execute("SELECT snapshot_id, base_snapshot_id FROM chat_fixture_metadata"):
                source = [row for row in old_bindings if row["snapshot_id"] == fixture["base_snapshot_id"]]
                current_product = chat.execute("SELECT product_id, product_name FROM products WHERE snapshot_id = ?",
                                               (fixture["snapshot_id"],)).fetchone()
                current_sku = chat.execute("SELECT product_id, sku_id FROM skus WHERE snapshot_id = ?",
                                           (fixture["snapshot_id"],)).fetchone()
                for row in source:
                    if (current_product is None or current_sku is None or
                        current_product["product_id"] != row["product_id"] or
                        current_product["product_name"] != row["product_name"] or
                        current_sku["product_id"] != row["product_id"] or
                        current_sku["sku_id"] != row["sku_id"]):
                        raise ValueError("chat fixture product/SKU differs from source binding")
                    derived.append({"binding_key": key(fixture["snapshot_id"], row["product_id"], row["sku_id"]),
                                    "snapshot_id": fixture["snapshot_id"], "product_id": row["product_id"],
                                    "sku_id": row["sku_id"], "product_name": row["product_name"],
                                    "manual_id": row["document_id"], "source_tier": "legacy_category",
                                    "match_basis": "verified_chat_fixture_inheritance",
                                    "base_snapshot_id": fixture["base_snapshot_id"]})
    bindings.extend(derived)

    cards = []
    for snapshot_id, row in products.items():
        fields = {name: row[name] for name in ("product_name", "material", "spec", "size")
                  if row[name] not in (None, "", "unknown")}
        if not row["product_id"] or not fields:
            continue
        cards.append({"card_id": f"FACT::{snapshot_id}", "snapshot_id": snapshot_id,
                      "product_id": row["product_id"], "sku_id": skus.get(snapshot_id, {}).get("sku_id"),
                      "fields": fields, "source_tier": "structured_product_fact",
                      "is_manual": False})

    queries = []
    for row in read_jsonl(GOLD):
        manual = frozen_by_id.get(row["manual_id"])
        if manual is None:
            raise ValueError("Gold references unknown frozen manual")
        binding = manual["product_binding"]
        gold_ids = [f"{row['manual_id']}::{section_id}" for section_id in row["gold_chunk_ids"]]
        queries.append({"query_id": row["query_id"], "user_query": row["user_query"],
                        "binding_key": key(binding["snapshot_id"], binding["product_id"], binding["sku_id"]),
                        "manual_id": row["manual_id"], "gold_chunk_ids": gold_ids,
                        "split": row["split"], "information_need_id": row["information_need_id"],
                        "label_quality": "human_authored_frozen_v1"})
    known_chunks = {row["chunk_id"] for row in chunks}
    if any(not set(row["gold_chunk_ids"]).issubset(known_chunks) for row in queries):
        raise ValueError("Gold references missing chunk")

    output_dir.mkdir(parents=True, exist_ok=True)
    files = {"manuals": manuals, "chunks": chunks, "bindings": bindings,
             "product_fact_cards": cards, "human_queries": queries}
    for name, rows in files.items():
        write_jsonl(output_dir / f"{name}.jsonl", rows)
    manifest = {"version": "unified_manual_corpus_v2", "source_files": {
        "legacy_sqlite": {"path": str(db_path.relative_to(ROOT)), "sha256": digest(db_path)},
        "frozen_manuals": {"path": str(frozen_manual_path.relative_to(ROOT)), "sha256": digest(frozen_manual_path)},
        "frozen_chunks": {"path": str(frozen_chunk_path.relative_to(ROOT)), "sha256": digest(frozen_chunk_path)},
        "human_queries": {"path": str(GOLD.relative_to(ROOT)), "sha256": digest(GOLD)},
    }, "counts": {**{name: len(rows) for name, rows in files.items()},
                  "unique_binding_keys": len({row["binding_key"] for row in bindings})},
        "source_counts": {"legacy_manuals": len(old_docs), "frozen_manuals": len(frozen),
                          "legacy_bindings": len(old_bindings), "frozen_bindings": len(frozen),
                          "verified_derived_bindings": len(derived)},
        "output_sha256": {f"{name}.jsonl": digest(output_dir / f"{name}.jsonl") for name in files},
        "note": "Synthetic research data. Old category and frozen product manuals remain distinct; product fact cards are not manuals."}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2,
                                                    sort_keys=True) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--chat-db", type=Path, default=CHAT_DB)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.db, args.chat_db, args.output)["counts"], ensure_ascii=False))
