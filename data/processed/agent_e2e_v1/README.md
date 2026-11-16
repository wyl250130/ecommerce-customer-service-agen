# Agent 全链路事实场景 v1

本目录用于核对**同一个模拟事实快照**中的订单、商品、SKU、类别说明书章节与参考回复。它不是现实商城交易记录，也不是真实厂商说明书。

## 两种场景

1. `scenarios.jsonl` 有 52 条可重建的一致性场景：51 条来自原库的 train 快照，1 条来自已验收的聊天派生快照。每条都有同快照订单、商品、SKU、原有精确说明书绑定、所选章节、问题和自动生成的参考回复。参考回复只用于字段一致性检查，**不是**人工答案 Gold 或真实模型输出。旧类别说明不包含特定 SKU 的全部参数，回复明确标注此边界。`tests/test_agent_e2e_scenarios.py` 对 52 条逐一调用 Agent 内部说明书工具验证精确章节。应用配置 `CS_MANUAL_SOURCE=legacy_sqlite` 后从原 SQLite 读取这些说明，不读取 RAG 实验集导出文件。
2. `../customer_service/customer_service_agent_full_chain.sqlite` 是独立 SQLite 副本：取冻结 v1 中“婴儿细轴棉签”的原商品快照，只新增一笔明确标记的模拟订单 `E2E-MANUAL-551-001`。该商品、SKU、冻结说明书和 6 个章节原封不动；原始训练库未修改。`agent/tests/test_full_chain_fixture.py` 使用脚本模型实际运行 LangGraph 的 `get_order_info → get_product_info → search_product_manual → finish_collection → LLM B` 路径，并检查最终回复与证据。它验证的是**脚本模型的执行链路**，不是外部 LLM 的回答质量。

## 重建和验证

在 `customer-service/` 目录运行：

```bash
python3 scripts/build_agent_full_chain_fixture.py
python3 -m scripts.build_agent_e2e_scenarios
python3 -m unittest tests.test_agent_e2e_scenarios -v
uv run python -m unittest agent.tests.test_full_chain_fixture -v
```

对应的来源哈希见 `manifest.json` 和 `../customer_service/customer_service_agent_full_chain.json`。这两套场景分开使用：51 条 train 场景用于多样的订单与旧类别说明绑定检查；隔离副本用于现有冻结说明书工具的完整 Graph 演练。不要把旧类别说明中的 236 条绑定当成 236 份独立说明书，也不要把新增的模拟订单声称为原 Gold 事实。

聊天演示场景另由 `agent/tests/test_full_chain_fixture.py` 使用 `LegacyManualSearchService` 跑通 `get_order_info → get_product_info → search_product_manual → finish_collection → LLM B`。这条路径的说明书内容来自原 SQLite，聊天派生快照的绑定仅在商品 ID、名称和 SKU 与来源快照完全一致时继承。运行的是脚本模型，仍未验证真实 LLM 的实际工具选择。

现有接口 `search_product_manual(query)` 是 Agent 内部工具，不是独立 HTTP 接口。用户可通过聊天 API 提问，由 LLM A 选择调用。测试 52 条旧类别说明时，在服务端设置 `CS_ENABLE_MANUAL_SEARCH=true`、`CS_MANUAL_SOURCE=legacy_sqlite`，并将 `CS_FACTS_DATABASE_PATH` 指向相应 SQLite；当前聊天演示快照仍由 `CS_DEFAULT_SNAPSHOT_ID=CHAT-E2E-SNAPSHOT-001` 选择。原冻结 v1 路径仍可用 `CS_MANUAL_SOURCE=frozen_v1`，需要精确冻结绑定和 Cross-Encoder 依赖。**本阶段没有做真实 LLM 的 HTTP 说明书验收。**
