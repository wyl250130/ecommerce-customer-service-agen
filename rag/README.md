# 商品说明书 RAG Retriever Baseline v1

这一版只评估：商品已经正确绑定后，Retriever 能否在该商品的 6 个说明书章节中召回 Retriever Gold v4 标注的证据。

## 当前后端

默认后端是完全离线可复现的中文字符 1～3 gram TF-IDF 向量。它用于验证 Indexer、商品过滤、余弦排序和评测链路，不是预训练语义向量模型。代码同时支持 `sentence_transformer` 后端，后续可在 GPU 服务器上用相同 Gold 做公平对比。

## 运行

在项目根目录执行：

```bash
python3 -m rag.indexer
python3 -m rag.retriever \
  --product-id 43650034536 \
  --query "浴巾脱线了还能继续用吗" \
  --top-k 3
python3 -m rag.evaluate
python3 -m unittest tests.test_rag_baseline -v
```

本机 BGE 实验复用了 Conda 环境 `transformer_env310`。关键依赖版本记录在 `rag/requirements-bge.txt`。建立独立 BGE 索引（不能覆盖 TF-IDF 基线）：

```bash
HF_HOME=.cache/huggingface \
/opt/miniconda3/envs/transformer_env310/bin/python -m rag.indexer \
  --backend sentence_transformer \
  --model-name BAAI/bge-small-zh-v1.5 \
  --output-dir data/rag_index/manual_v1_bge_small_zh

HF_HOME=.cache/huggingface HF_HUB_OFFLINE=1 \
/opt/miniconda3/envs/transformer_env310/bin/python -m rag.evaluate \
  --index-dir data/rag_index/manual_v1_bge_small_zh \
  --output-dir outputs/rag_retriever_baseline_v1_bge_small_zh

python3 -m rag.compare
```

## 产物

- `data/rag_index/manual_v1_tfidf_char/`：向量、metadata、向量器和索引清单。
- `outputs/rag_retriever_baseline_v1_tfidf_char/retrieval_metrics.json`：聚合指标。
- `outputs/rag_retriever_baseline_v1_tfidf_char/retrieval_results.jsonl`：48 条逐题结果。
- `outputs/rag_retriever_baseline_v1_tfidf_char/gold_empty_results.jsonl`：14 条 Gold-empty 观察结果，不进入主指标。
- `outputs/rag_retriever_baseline_v1_tfidf_char/retrieval_error_report.md`：失败和部分失败分析。
- `outputs/rag_retriever_baseline_v1_bge_small_zh/`：BGE 完整评测产物。
- `outputs/rag_retriever_comparison_v1/`：聚合差值、逐题变化与中文结论。

## 指标口径

- Recall@K：Top-K 是否至少包含一个 Gold chunk。
- Evidence Recall@K：Top-K 命中的 Gold chunk 数除以 Gold chunk 总数。
- MRR：第一个 Gold chunk 排名的倒数均值。
- Gold-empty 不进入以上指标，也不使用相似度阈值臆造 no-answer 判断。

## Handwritten v2 正式三基线

数据集为 160 个 information needs、800 条人工 query。完整复现顺序：

```bash
python3 scripts/compile_handwritten_retriever_data.py

python3 -m rag.indexer \
  --backend tfidf_char \
  --output-dir data/rag_index/manual_v1_tfidf_char

HF_HOME=.cache/huggingface HF_HUB_OFFLINE=1 \
/opt/miniconda3/envs/transformer_env310/bin/python -m rag.indexer \
  --backend sentence_transformer \
  --model-name BAAI/bge-small-zh-v1.5 \
  --output-dir data/rag_index/manual_v1_bge_small_zh

HF_HOME=.cache/huggingface HF_HUB_OFFLINE=1 \
/opt/miniconda3/envs/transformer_env310/bin/python -m rag.indexer \
  --backend sentence_transformer \
  --model-name BAAI/bge-small-zh-v1.5 \
  --query-prefix "为这个句子生成表示以用于检索相关文章：" \
  --output-dir data/rag_index/manual_v1_bge_small_zh_instruction

python3 -m rag.evaluate \
  --gold data/processed/retriever_eval/manual_v1_handwritten_v2_compiled.jsonl \
  --index-dir data/rag_index/manual_v1_tfidf_char \
  --output-dir outputs/retriever_handwritten_v2_final/tfidf_char

HF_HOME=.cache/huggingface HF_HUB_OFFLINE=1 \
/opt/miniconda3/envs/transformer_env310/bin/python -m rag.evaluate \
  --gold data/processed/retriever_eval/manual_v1_handwritten_v2_compiled.jsonl \
  --index-dir data/rag_index/manual_v1_bge_small_zh \
  --output-dir outputs/retriever_handwritten_v2_final/bge_small_zh

HF_HOME=.cache/huggingface HF_HUB_OFFLINE=1 \
/opt/miniconda3/envs/transformer_env310/bin/python -m rag.evaluate \
  --gold data/processed/retriever_eval/manual_v1_handwritten_v2_compiled.jsonl \
  --index-dir data/rag_index/manual_v1_bge_small_zh_instruction \
  --output-dir outputs/retriever_handwritten_v2_final/bge_small_zh_instruction

python3 -m rag.analyze_handwritten_v2
```

正式结论位于 `outputs/retriever_handwritten_v2_final/final_report.md`。统计显著性以 information need 为 bootstrap 单位，替代 Gold 按 `acceptable_gold_groups` 计算。
