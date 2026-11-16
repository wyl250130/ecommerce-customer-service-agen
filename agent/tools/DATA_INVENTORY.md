# 工具数据资产实测清单

统计口径：2026-10-28 本地数据库与文件。字节大小会随数据库维护变化；非空不代表业务事实已经确认。

数据库：`data/processed/customer_service/customer_service_train.sqlite`；文件大小 **2,043,904 bytes**。

## `after_sales`：445 行

| 字段 | SQLite 类型 | 主键 | NOT NULL | SQL NULL 数 |
| --- | --- | --- | --- | ---: |
| `snapshot_id` | TEXT | 是 |  | 0 |
| `order_id` | TEXT |  |  | 0 |
| `after_sales_status` | TEXT |  |  | 0 |
| `after_sales_request_status` | TEXT |  |  | 0 |
| `return_status` | TEXT |  |  | 0 |
| `refund_status` | TEXT |  |  | 0 |
| `refund_processing_status` | TEXT |  |  | 0 |
| `reshipment_status` | TEXT |  |  | 0 |
| `compensation_status` | TEXT |  |  | 444 |
| `extra_json` | TEXT |  | 是 | 0 |

## `fact_import_metadata`：1 行

| 字段 | SQLite 类型 | 主键 | NOT NULL | SQL NULL 数 |
| --- | --- | --- | --- | ---: |
| `source_path` | TEXT | 是 |  | 0 |
| `sha256` | TEXT |  | 是 | 0 |
| `imported_at` | TEXT |  | 是 | 0 |

## `fact_snapshots`：625 行

| 字段 | SQLite 类型 | 主键 | NOT NULL | SQL NULL 数 |
| --- | --- | --- | --- | ---: |
| `snapshot_id` | TEXT | 是 |  | 0 |
| `sample_id` | TEXT |  | 是 | 0 |
| `store_id` | TEXT |  |  | 65 |
| `store_name` | TEXT |  |  | 65 |
| `fact_status` | TEXT |  |  | 0 |
| `evidence_status` | TEXT |  |  | 0 |
| `evidence_source` | TEXT |  |  | 0 |
| `valid_until` | TEXT |  |  | 0 |
| `raw_fact_json` | TEXT |  | 是 | 0 |

## `logistics`：445 行

| 字段 | SQLite 类型 | 主键 | NOT NULL | SQL NULL 数 |
| --- | --- | --- | --- | ---: |
| `snapshot_id` | TEXT | 是 |  | 0 |
| `order_id` | TEXT |  |  | 0 |
| `tracking_no` | TEXT |  |  | 427 |
| `actual_carrier` | TEXT |  |  | 353 |
| `logistics_status` | TEXT |  |  | 0 |
| `estimated_delivery` | TEXT |  |  | 443 |
| `arrival_eta` | TEXT |  |  | 444 |
| `dispatch_commitment` | TEXT |  |  | 0 |
| `extra_json` | TEXT |  | 是 | 0 |

## `orders`：445 行

| 字段 | SQLite 类型 | 主键 | NOT NULL | SQL NULL 数 |
| --- | --- | --- | --- | ---: |
| `snapshot_id` | TEXT | 是 |  | 0 |
| `order_id` | TEXT |  |  | 0 |
| `order_record_status` | TEXT |  |  | 0 |
| `order_item_status` | TEXT |  |  | 0 |
| `payment_status` | TEXT |  |  | 0 |
| `shipment_status` | TEXT |  |  | 0 |
| `address_status` | TEXT |  |  | 440 |
| `address_change_status` | TEXT |  |  | 442 |
| `order_change_status` | TEXT |  |  | 444 |
| `extra_json` | TEXT |  | 是 | 0 |

## `products`：343 行

| 字段 | SQLite 类型 | 主键 | NOT NULL | SQL NULL 数 |
| --- | --- | --- | --- | ---: |
| `snapshot_id` | TEXT | 是 |  | 0 |
| `product_id` | TEXT |  |  | 0 |
| `product_name` | TEXT |  |  | 12 |
| `material` | TEXT |  |  | 229 |
| `spec` | TEXT |  |  | 138 |
| `size` | TEXT |  |  | 317 |
| `product_match` | INTEGER |  |  | 0 |
| `stock_status` | TEXT |  |  | 0 |
| `extra_json` | TEXT |  | 是 | 0 |

## `rule_documents`：19 行

| 字段 | SQLite 类型 | 主键 | NOT NULL | SQL NULL 数 |
| --- | --- | --- | --- | ---: |
| `policy_id` | TEXT | 是 |  | 0 |
| `title` | TEXT |  |  | 0 |
| `rule_type` | TEXT |  |  | 0 |
| `normalized_rule` | TEXT |  |  | 0 |
| `source_level` | TEXT |  |  | 0 |
| `review_status` | TEXT |  |  | 0 |
| `source_url` | TEXT |  |  | 12 |
| `raw_rule_json` | TEXT |  | 是 | 0 |

## `skus`：331 行

| 字段 | SQLite 类型 | 主键 | NOT NULL | SQL NULL 数 |
| --- | --- | --- | --- | ---: |
| `snapshot_id` | TEXT | 是 |  | 0 |
| `sku_id` | TEXT |  |  | 0 |
| `product_id` | TEXT |  |  | 0 |
| `price_text` | TEXT |  |  | 294 |
| `stock_status` | TEXT |  |  | 0 |
| `purchase_limit` | TEXT |  |  | 0 |
| `extra_json` | TEXT |  | 是 | 0 |

## `usage_document_bindings`：236 行

| 字段 | SQLite 类型 | 主键 | NOT NULL | SQL NULL 数 |
| --- | --- | --- | --- | ---: |
| `binding_id` | INTEGER | 是 |  | 0 |
| `document_id` | TEXT |  | 是 | 0 |
| `snapshot_id` | TEXT |  | 是 | 0 |
| `product_id` | TEXT |  |  | 0 |
| `sku_id` | TEXT |  |  | 0 |
| `product_name` | TEXT |  |  | 0 |
| `match_basis` | TEXT |  | 是 | 0 |

## `usage_document_metadata`：8 行

| 字段 | SQLite 类型 | 主键 | NOT NULL | SQL NULL 数 |
| --- | --- | --- | --- | ---: |
| `document_id` | TEXT | 是 |  | 0 |
| `version` | TEXT |  | 是 | 0 |
| `effective_time` | TEXT |  | 是 | 0 |
| `review_status` | TEXT |  | 是 | 0 |
| `is_simulated` | INTEGER |  | 是 | 0 |
| `warnings_json` | TEXT |  | 是 | 0 |
| `raw_document_json` | TEXT |  | 是 | 0 |

## `usage_documents`：8 行

| 字段 | SQLite 类型 | 主键 | NOT NULL | SQL NULL 数 |
| --- | --- | --- | --- | ---: |
| `document_id` | TEXT | 是 |  | 0 |
| `product_id` | TEXT |  |  | 8 |
| `title` | TEXT |  | 是 | 0 |
| `content` | TEXT |  | 是 | 0 |
| `source` | TEXT |  | 是 | 0 |

## 规则审核状态

- `approved_for_rule_context_with_temporal_note`：1 条
- `approved_for_simulation`：12 条
- `pending`：6 条

## RAG 文件规模

### `data/processed/product_manuals/frozen_v1/product_manuals_frozen_v1.jsonl`

- 记录数：8
- 文件大小：20,845 bytes
- SHA-256：`b8a469b78e552262e6ff194c2348310434d9a18256fd5be7f76f25686633ffcf`
- 字段并集：`content_sha256`、`document_type`、`frozen_at`、`gold_hard_constraints`、`is_synthetic`、`knowledge_boundary`、`manual_id`、`product_binding`、`sections`、`selection_reason`、`source`、`status`、`version`

### `data/processed/product_manuals/frozen_v1/product_manual_chunks_frozen_v1.jsonl`

- 记录数：48
- 文件大小：34,846 bytes
- SHA-256：`3d2c5585088d355cd75120a03888a20709fce4f087ac03b6a6cf52891756500a`
- 字段并集：`chunk_id`、`chunk_sha256`、`content`、`manual_content_sha256`、`manual_id`、`manual_version`、`product_id`、`product_name`、`section_id`、`section_title`、`sku_id`、`snapshot_id`、`source`

### `data/processed/simulated_knowledge/simulated_usage_manuals_v1.jsonl`

- 记录数：8
- 文件大小：9,781 bytes
- SHA-256：`723f49c4f5680007c1e1990e3d4a9d3a540e8f3519b7ccfd72d93f856bf4d961`
- 字段并集：`category`、`content`、`document_id`、`document_type`、`effective_time`、`is_simulated`、`review_status`、`source`、`title`、`version`、`warnings`

### `data/processed/official_rules/retrieval_v1/policy_chunks_v1.jsonl`

- 记录数：24
- 文件大小：115,079 bytes
- SHA-256：`f6fe0188fb7829ab9d1e266d5077ae9bbe5f0b6bb7543f06887a98cbfdd1c5c7`
- 字段并集：`applicability`、`article_range`、`authority_tier`、`chunk_id`、`chunk_review_status`、`chunk_text`、`chunk_title`、`conditions`、`customer_service_relevant`、`document_role`、`evidence_required`、`excerpt_mode`、`must_mention`、`must_not_say`、`online_eligible`、`parent_policy_id`、`parent_title`、`policy_version`、`retrieval_ready`、`retrieval_text`、`scope`、`source_spans`、`source_text_sha256`、`source_url`、`status`、`triggers`
- `chunk_review_status`：{'pending_review': 24}
- `retrieval_ready`：{'False': 24}

### `data/processed/official_rules/governance_v2/active_policy_corpus_v1.jsonl`

- 记录数：6
- 文件大小：72,864 bytes
- SHA-256：`51afbf45cac1b63bbbb40fabcee2bb087b50716fe506f742ffa5d6016accefda`
- 字段并集：`applicability`、`authority_tier`、`conditions`、`customer_service_relevant`、`document_role`、`domain`、`escalation_when`、`evidence_required`、`must_mention`、`must_not_say`、`normalized_rule`、`online_eligible`、`parent_policy_id`、`policy_family_id`、`policy_id`、`policy_version`、`published_at_or_source_time`、`requires_chunking`、`retrieval_ready`、`review`、`rule_text`、`scope`、`source_line`、`source_text_sha256`、`source_url`、`status`、`title`、`triggers`
- `retrieval_ready`：{'False': 6}
