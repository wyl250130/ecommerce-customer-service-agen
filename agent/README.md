# Agent Phase 1 Local Simulation

工具整理后的总入口见 [业务工具总README](tools/README.md)：目录、每个工具输入输出、全部表字段、数据规模与RAG检索方式。原生工具调用System Prompt位于 `prompts/agent_tool_system_v1.txt`；schema和桥接位于 `agent/tool_calling.py`。本文后续阶段记录中的旧文件名仍保留为兼容导入。

本目录实现第一阶段本地客服 Agent 模拟环境。当前目标不是上线客服后端，而是验证：

```text
SessionState
  -> ToolRegistry
  -> get_order_info / get_logistics
  -> SQLite
  -> Evidence
  -> EvidenceGate
  -> Runner / Recorder / Verifier
```

## 当前已完成

- `schemas.py`：统一 `ToolAction`、`ToolResult`、`Evidence`、`GateResult`。
- `session_state.py`：保存会话状态、已获得证据、工具调用历史。
- `tools/order.py`：`get_order_info`，按 `snapshot_id + order_id` 查询订单。
- `tools/logistics.py`：`get_logistics`，按 `snapshot_id + order_id` 查询物流。
- `tools/business.py`：新增 `get_recent_orders`、`get_product_info`、`get_after_sales`，复用现有订单、商品、SKU和售后表。
- `tools/manual.py`：实验性 `search_product_manual` 薄封装，对已锁定商品的 6 个冻结章节使用 Direct Cross-Encoder 排序。
- `tool_registry.py`：工具注册与安全执行；禁止 Agent 自己传 `snapshot_id`。
- `evidence.py`：从当前状态收集字段。
- `evidence_gate.py`：确定性判断发货/物流查询证据是否充分。
- `runner.py`：手工 action 执行循环，包含最大步数、重复调用、未知工具和参数错误保护。
- `recorder.py`：记录每一步 `state_before -> action -> observation -> state_after`，避免把 observation 泄漏进动作训练输入。
- `verifier.py`：确定性检查轨迹是否存在未知工具、跨 snapshot、重复调用和过早 finish。

## 当前未接入

- LLM A / Teacher Agent。
- Qwen3-4B 最终回复模型。
- 待审规则chunk的正式RAG检索（旧SQLite关键词原型 `search_policies` 已接入）。
- 说明书问答的证据充分性判断、自动拒答和最终回复生成（目前仅完成检索工具适配）。
- 多订单消歧和模拟用户追问。

## 新增业务工具（2026-10-28）

事实数据现在统一使用 `final_semantic_v3_consistency/train_sft_gold.jsonl`（625条），不混入 dev/test。
`scripts/sync_customer_service_facts.py` 在现有数据库中更新六张事实表，保留规则和说明书，写入来源哈希，并在同步前创建完整 SQLite 备份。建库脚本的默认来源也已改为 v3 train。

| 工具 | 模型参数 | 结果 |
| --- | --- | --- |
| `get_recent_orders` | 可选 `limit`（1～20，默认5）、`product_hint` | 当前应用授权快照范围内的订单摘要、商品/SKU、付款/发货状态 |
| `get_product_info` | `product_id`、`sku_id` 至少一个；同时给出时两者必须匹配 | 商品属性、SKU、价格文本、库存、限购及各自扩展信息 |
| `get_after_sales` | `order_id` | 售后、申请、退货、退款处理、补发、补偿状态及扩展信息 |

`get_recent_orders` 的范围由应用设置 `SessionState.visible_snapshot_ids`。默认只查询当前快照；传空列表表示没有可见订单，绝不回退全库。该范围是演示设置，不是从数据集推断的用户订单归属。原数据没有下单时间，因此返回 `order_time=null`、`ordering=application_scope_not_chronological`、`ownership_verified=false`，状态为 `partial`；不得对用户声称这是按时间排序的最近订单。商品提示使用字面包含匹配，不承担语义消歧。

候选中的订单号可用于后续订单、物流和售后查询。执行器只在授权范围中定位，切换目标时清除旧对象证据，避免串单。订单查询成功后从同一快照绑定商品和 SKU；`get_product_info` 成功结果经 `state.add_tool_result` 保存后也会更新商品绑定，可供说明书工具使用。

```python
import sqlite3
from agent.schemas import ToolAction
from agent.session_state import SessionState
from agent.tool_registry import execute_tool
from agent.tools.common import configure_connection

conn = configure_connection(sqlite3.connect(
    "data/processed/customer_service/customer_service_train.sqlite"))
state = SessionState("demo", "GOLD-SNAP-SEMANTIC-GOLD-TRAIN-000140")
# 多订单演示时由应用指定实际存在的快照，不能让模型填写此参数。
state.visible_snapshot_ids = [state.snapshot_id]
action = ToolAction("get_recent_orders", {"limit": 5})
result = execute_tool(action, state, conn)
state.add_tool_result(action, result)
print(result.to_dict())
conn.close()
```

这三个工具仅完成查询和证据返回。LLM A、自然语言澄清、LangGraph、最终答复与通用动作校验方案仍需后续接入；旧 `EvidenceGate` 仍仅支持发货/物流意图。

这些将在核心工具闭环稳定后继续接入。

## 运行测试

```bash
python3 -m unittest discover -s agent/tests -v
```

## LangGraph 工具适配层

`tools/` 中的 SQL 和说明书查询函数保持原样。`tool_adapters/` 提供 10 个模型可调用的 LangChain 工具；`@tool` 只向模型暴露业务参数，`ToolRuntime[RuntimeContext]` 注入可信数据库范围和 Graph State。`ToolNode` 执行工具；适配器用 `Command` 写入 `ToolMessage`、当前订单和 `evidence_generation`。工具调用历史只从 `messages` 中的 `AIMessage.tool_calls` 与 `ToolMessage` 提取。`finish_collection` 使用 `evidence.py` 的当前代次证据顺序校验索引。

首次绑定订单或切换订单时，`evidence_generation` 都会递增；旧 `ToolMessage` 留在消息历史中，但不再作为当前证据。规则检索结果也会失效，因为规则是否适用可能依赖所选订单；需要时应在选定订单后重新检索规则。

新入口是 `build_agent(llma_model, llmb_model)`，调用时传入 `context=RuntimeContext(...)`，并用 `make_initial_state()` 一次性设置 `tool_call_count=0`、`evidence_generation=0` 等初始字段。Graph 入口会先校验这两个字段，缺少时直接报错，避免 Adapter 读到未初始化的 State。SQLite 工具由 `ToolNode` 在线程中执行，因此 Context 只提供 `connection_factory`：每次调用在执行线程中创建连接，结束后关闭。`ManualSearchService` 和 `HandoffClient` 在应用启动时创建并通过 Context 复用。`conversation_id` 用于人工客服会话关联；未来如启用 Checkpointer，其 `thread_id` 仍由应用层单独传入。例如：

```python
import sqlite3
from agent.context import RuntimeContext
from agent.graph import build_agent
from agent.state import make_initial_state
from agent.tools.common import configure_connection

context = RuntimeContext(
    connection_factory=lambda: configure_connection(sqlite3.connect(db_path)),
    default_snapshot_id=snapshot_id,
    visible_snapshot_ids=(snapshot_id,),
    conversation_id=conversation_id,
    handoff_client=handoff_client,
)
graph = build_agent(llma_model, llmb_model)
initial_state = make_initial_state(user_messages, context=context)
result = graph.invoke(
    initial_state,
    context=context,
)
```

旧的 `build_agent(..., conn, session_context, handoff_client)` 接口仍可用于已有本地模拟。

## 说明书检索工具（阶段 A）

后续补充：完整客服工程现可用 `CS_MANUAL_SOURCE=legacy_sqlite` 选择原 SQLite 的类别级模拟说明，服务端仍按快照、商品、SKU 精确绑定。这个模式不读取离线 RAG 实验集；结果明确标记为类别建议，不能冒充厂商专属说明。下文描述的是保留的 `frozen_v1` 模式。

应用启动时创建一次 `ManualSearchService` 并传给 `AgentRunner(..., manual_service=service)`；服务会加载冻结说明书、章节和 reranker，后续会话复用，不在每条请求中重载。模型的调用参数只能是 `{"query":"用户的问题"}`。应用层须先在 `SessionState` 里确定 `snapshot_id`、`current_product_id`、`current_sku_id`；模型不能选择商品、SKU、快照、说明书或 Top-K。

第一版仅允许冻结说明书中登记的 8 组精确 `snapshot_id + product_id + sku_id` 绑定；相同商品出现在其他快照时也不会自动共享说明书。尚未确认商品或 SKU 返回 `partial`，无合法绑定返回 `no_match`，推理故障返回 `tool_error`。若需要跨快照复用，必须先建立并审核显式绑定。

默认返回 Top-5；实验配置可固定为 3、5、6。结果保存章节原文、顺序、原始 reranker 分数、说明书 ID/版本、章节 ID、chunk 哈希和语料文件 SHA-256。服务启动时核对冻结 manifest、固定语料哈希、两份冻结文件的固定 SHA-256，以及每份说明书/章节的内容哈希；即使改动章节后同步重算其自带哈希，也会因文件哈希不符而拒绝加载。改动文件需发布新的受控版本。执行器允许 `partial` 后继续追问、`no_match` 后继续转人工；验证器识别说明书工具并拒绝模型传入范围参数。分数不是概率，`found` 不代表问题有答案或证据足以回复。说明书意图的证据充分性检查和完整来源验证属于后续阶段；当前不得把工具检索成功当成自动 `finish` 许可。

本地运行需要安装 `torch`、`transformers`，并可设置 Hugging Face 缓存。无模型的边界测试通过注入模拟 scorer 运行，不依赖网络。

## 运行 10 条手工 rollout

```bash
python3 scripts/run_phase1_manual_rollout.py --limit 10
```

输出：

```text
experiments/agent_phase1/manual_rollout_10.jsonl
```

该文件保存每条样本的工具调用、工具 observation、证据门结果和 verifier 结果，可作为后续 Agent trajectory SFT 的原始轨迹格式参考。

## 关键边界

- Agent 不能直接访问数据库。
- Agent 不能看到完整 `backend_world`。
- `snapshot_id` 永远由 `SessionState` 注入，不能由模型传入。
- `unknown`、`null`、`not_applicable` 不应被模型猜成确定事实。
- `Evidence` 不一定是正面事实；`not_found`、`error`、`conflict` 也是有效 observation，但不能直接支持最终回答。
- `EvidenceGate` 区分字段不存在与字段存在但值未知：前者进入 `missing`，后者进入 `unknown`。
- 当前只验证发货/物流查询纵向链路，不代表完整客服系统已经完成。
