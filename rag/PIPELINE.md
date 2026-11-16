# 检索流水线

`base.py` 规定统一的 `search(product_id, query, top_k)` 接口。`schemas.py` 规定结果字段，`pipeline.py` 组合召回与可选重排，`factory.py` 根据应用配置创建检索器。原有 `direct_cross_encoder.py`、`retriever.py`、索引和评测脚本仍可单独使用。

可选策略：

- `direct_cross_encoder`：对已绑定商品的全部冻结章节直接打分；当前 Agent 默认策略。
- `bm25`：在已加载的冻结章节中做字符 bigram BM25 检索，无额外模型依赖。
- `dense`：包装原有 `ManualRetriever`，需要与语料对应的 `index_dir`；具体编码器由索引清单决定。
- `hybrid`：BM25 与 `dense` 的结果做倒数排名融合，需要 `chunks` 和 `index_dir`。

可选 `reranker="cross_encoder"`，需提供评分器或允许工厂加载默认模型。`candidate_k` 决定送入重排的候选数，至少应等于最终 `top_k`。

```python
from pathlib import Path
from agent.tools.manual import ManualSearchService

# 应用启动时创建一次，在会话中复用。
manual_service = ManualSearchService(
    retrieval_strategy="hybrid",
    index_dir=Path("data/rag_index/manual_v1_bge_small_zh"),
    reranker="cross_encoder",
    candidate_k=6,
    top_k=5,
)
```

应用配置可先对应到上述构造参数；切换配置时重新创建 `ManualSearchService`，再把新实例交给 `RuntimeContext`。LLM 的工具参数仍只有 `query`，商品、SKU 和快照由应用及 Agent 状态绑定。

`ManualSearchService` 继续核验冻结语料的哈希及商品/SKU 绑定，返回结果时从冻结章节读取正文与来源信息。新建或更换 dense 索引时，应先验证索引与冻结章节一致，再用于正式服务。
