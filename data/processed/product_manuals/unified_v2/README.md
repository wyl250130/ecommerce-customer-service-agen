# 说明书统一检索实验集 v2

这是**研究用模拟数据**的只读导出，不替换旧 SQLite、冻结 v1 语料或人工 Gold。重新生成：

```bash
cd customer-service
python3 scripts/build_manual_corpus_v2.py
```

| 文件 | 用途 |
| --- | --- |
| `manuals.jsonl` | 8 份旧类别说明 + 8 份冻结商品说明，保留 `source_tier` |
| `chunks.jsonl` | 旧说明按原有语句切块 + 冻结 v1 原章节，共 104 块 |
| `bindings.jsonl` | `snapshot_id + product_id + sku_id` 到说明书的精确绑定，共 245 条记录、239 个不同绑定键 |
| `product_fact_cards.jsonl` | 从旧库商品表只读导出的 334 张有内容的商品事实卡；**不是说明书** |
| `human_queries.jsonl` | 原手写 v2 的 800 条问题，Gold 章节 ID 转为本语料的全局 ID |
| `manifest.json` | 来源与输出 SHA-256、数量；检索器加载时核对输出哈希 |

245 条绑定由旧库 236 条、冻结 v1 的 8 条，以及聊天派生快照中 1 条核对过商品和 SKU 的继承绑定组成。**绑定数量不是独立说明书数量。**旧类别说明只提供类别级模拟建议；冻结商品说明是商品和 SKU 专属内容。相同绑定同时命中两种来源时，默认只返回冻结商品说明；实验时可显式包含旧说明作对照。不能凭商品名相似自动跨快照绑定。

旧数据的 8 份类别说明来自 `data/processed/customer_service/customer_service_train.sqlite`；冻结 v1 的 8 份说明来自 `data/processed/product_manuals/frozen_v1/`。手写 v2 的原始编译文件按字节复制到 `data/processed/retriever_eval/manual_v1_handwritten_v2_compiled.jsonl`，使这个独立项目可重建；哈希与上级研究目录原件一致。原数据、哈希与旧评测结果均未改动。聊天派生快照只在 `chat_fixture_metadata.base_snapshot_id` 指向已有旧绑定，且派生商品 ID、名称、SKU 与来源完全一致时继承绑定。

## 检索与对比

```bash
python3 -m rag.unified_manuals \
  --snapshot-id CHAT-E2E-SNAPSHOT-001 \
  --product-id context-product-GOLD-DEV-000063 \
  --sku-id snack-durian \
  --query '冻干榴莲开封后怎么保存？' --top-k 3

python3 -m rag.evaluate_unified_manuals --split dev
python3 -m unittest tests.test_unified_manual_corpus -v
```

`UnifiedManualCorpus.candidates(snapshot_id, product_id, sku_id)` 返回严格限定的章节，可交给 BM25、TF-IDF、向量、重排模型等不同策略；`search()` 只是无需额外依赖的中文字符 BM25 基线。若直接比较自己的排序结果，按 JSONL 输出 `{"query_id":"...","retrieved_chunk_ids":["MANUAL-ID::section",...]}`，然后运行 `python3 -m rag.evaluate_unified_manuals --split dev --predictions 结果.jsonl`。输出的章节必须来自该问题的精确绑定。默认只评 dev；test 应在策略冻结后再使用。Gold 为空的题单独计数，不进入有答案题的召回指标。

`human_queries.jsonl` **只标注冻结 v1 的 8 份商品说明**。旧类别说明的 236 条绑定可用于绑定正确性、隔离和扩展覆盖测试，但没有人工章节 Gold，不能把它们计入正式 RAG 召回率。商品事实卡可用于商品定位或作为负例候选；不能把它们冒充说明书正文。16 份独立说明书仍不足以证明大规模跨商品检索泛化，后续扩充需要新增独立撰写并审核的说明书及章节标注。

当前 HTTP Agent 仍使用原 `ManualSearchService` 的冻结 v1 入口；这个导出集是离线 RAG 对比入口。确定检索策略和人工审核规则后，再把统一候选访问接入在线工具。
