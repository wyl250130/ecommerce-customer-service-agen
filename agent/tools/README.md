# 电商客服业务工具总说明

更新：2026-10-28。本文是工具入口文档；字段类型、空值数量、文件字节数和SHA-256详见 [数据实测清单](DATA_INVENTORY.md)。所有业务工具均为只读查询，不执行退款、补发、支付或订单修改。

## 1. 文件目录与职责

```text
agent/
  tools/
    orders/info.py         get_order_info
    orders/recent.py       get_recent_orders
    shipping/logistics.py  get_logistics
    products/info.py       get_product_info
    after_sales/status.py  get_after_sales
    policies/keyword.py    原关键词排序算法
    policies/search.py     search_policies 的统一结果封装
    manuals/search.py      search_product_manual / ManualSearchService
    common.py              SQLite连接、行转换和未知值辅助
    business_common.py     新业务工具共享结果辅助
    order.py, logistics.py, manual.py, business.py
                           旧导入路径兼容层；不重复实现业务
    README.md              本文
    DATA_INVENTORY.md      实测表结构和语料规模
  tool_registry.py          注册、参数范围保护、候选选择与执行
  tool_calling.py           唯一工具schema、原生与LangGraph调用分发
  handoff.py                转人工原因码、接口协议与本地模拟客服系统
  nodes/handoff.py          Handoff Node：请求转接并更新service_status
  state.py                  LangGraph消息、调用次数与转人工状态
  session_state.py          会话对象、工具历史、证据与对象绑定
  schemas.py                ToolResult等内部类型
  runner.py                 旧手工动作执行器（尚非LangGraph）
  evidence.py, evidence_gate.py, verifier.py, recorder.py
                           旧证据门、轨迹记录和校验
  tests/                    查询、隔离、说明书边界和原生调用桥接测试
prompts/agent_tool_system_v1.txt
                           给LLM A的工具使用System Prompt
scripts/build_customer_service_db.py
                           从v3 train全量新建数据库（覆盖会丢失后导入说明书）
scripts/sync_customer_service_facts.py
                           备份后同步现有六张事实表，保留规则和说明书
scripts/build_simulated_knowledge.py
                           旧模拟说明书及236条绑定的构建脚本
scripts/build_frozen_product_manuals_v1.py
                           冻结说明书构建，禁止原地编辑冻结版本
scripts/search_simulated_knowledge.py
                           旧CLI入口；规则部分复用policies/keyword.py
scripts/inspect_business_tool_assets.py
                           只读输出数据库与语料实测报告
rag/direct_cross_encoder.py Cross-Encoder底层评分和商品过滤
rag/indexer.py, retriever.py, evaluate*.py
                           说明书检索实验与离线评测，不是额外业务工具
```

数据和冻结语料保留原目录，避免破坏脚本、哈希和外部路径引用。每个业务目录拥有对应实现；数据文件由统一数据目录管理，不能为每个工具复制一份数据库。

## 2. 当前完成范围

7个业务工具均已注册：`get_recent_orders`、`get_order_info`、`get_logistics`、`get_product_info`、`get_after_sales`、`search_policies`、`search_product_manual`。

规则工具本次接入的是已有SQLite规则卡关键词原型，不代表待审24个规则chunk已经上线。说明书工具需启动时提供服务：`frozen_v1` 使用 `ManualSearchService`，`legacy_sqlite` 使用 `LegacyManualSearchService`。没有服务会返回 `MANUAL_SERVICE_UNAVAILABLE`。

当前没有接入真实LLM A、LLM B、前端、真实人工客服系统或完整LangGraph。已有可注入客服接口的Handoff Node，本地 `SimulatedHandoffClient` 仅生成模拟工单号，不联系真人。旧Runner只支持手工动作并保留发货/物流两个意图的确定性EvidenceGate。新的System Prompt和原生调用桥接供下一阶段图节点使用，不代表模型已经学会工具调用。

## 3. 数据来源及表规模

订单、商品、SKU、物流和售后事实统一来自 `data/processed/gold_sft/final_semantic_v3_consistency/train_sft_gold.jsonl`，625条训练样本；未将dev/test事实导入训练查询库。数据库为 `data/processed/customer_service/customer_service_train.sqlite`。

| 表 | 行数 | 用途/使用工具 |
| --- | ---: | --- |
| fact_snapshots | 625 | 完整原始事实、样本编号及可追溯证据 |
| products | 343 | 商品属性；商品查询和订单候选 |
| skus | 331 | SKU、价格文本、库存；商品查询和候选 |
| orders | 445 | 订单详情和候选查询 |
| logistics | 445 | 物流查询 |
| after_sales | 445 | 售后查询 |
| rule_documents | 19 | 规则卡；其中13条审核状态允许关键词检索 |
| usage_documents | 8 | 旧SQLite类别模拟说明，`legacy_sqlite` 模式读取 |
| usage_document_metadata | 8 | 旧文档版本和警告 |
| usage_document_bindings | 236 | 旧文档与235个商品快照绑定，`legacy_sqlite` 模式按快照、商品、SKU精确检索 |
| fact_import_metadata | 1 | 最近事实导入的来源路径、SHA-256及时间 |

主关联键是 `snapshot_id`：它限定同一份样本事实。商品ID和SKU不能跨快照默认合并；真实用户归属、订单商品数量和下单时间不存在。会话中的 `visible_snapshot_ids` 是应用配置的演示可见范围，不是数据集提供的用户订单关系。

### 表字段含义

字段的完整SQLite类型、主键和NULL统计在 [DATA_INVENTORY.md](DATA_INVENTORY.md) 中，以下解释业务含义。

| 表 | 字段说明 |
| --- | --- |
| fact_snapshots | `snapshot_id`快照主键；`sample_id`唯一样本号；`store_id/store_name`店铺；`fact_status/evidence_status`事实与证据状态；`evidence_source`来源；`valid_until`数据标记有效期；`raw_fact_json`原始完整事实 |
| products | `snapshot_id`主键兼外键；`product_id/product_name`商品；`material`材质；`spec`规格原文；`size`尺寸；`product_match`原样本是否匹配；`stock_status`库存状态；`extra_json`扩展属性 |
| skus | `snapshot_id`主键兼外键；`sku_id/product_id`规格及商品；`price_text`价格原文，不保证是数值；`stock_status`库存状态；`purchase_limit`限购原文；`extra_json`扩展信息 |
| orders | `snapshot_id`主键兼外键；`order_id`订单号；`order_record_status`记录匹配情况；`order_item_status`商品情况；`payment_status`付款；`shipment_status`发货；`address_status`地址；`address_change_status`改地址处理；`order_change_status`改单处理；`extra_json`扩展信息 |
| logistics | `snapshot_id`主键兼外键；`order_id`订单；`tracking_no`运单；`actual_carrier`承运商；`logistics_status`物流状态；`estimated_delivery/arrival_eta`预计到货原文；`dispatch_commitment`发货承诺；`extra_json`扩展信息 |
| after_sales | `snapshot_id`主键兼外键；`order_id`订单；`after_sales_status`售后总状态；`after_sales_request_status`是否申请；`return_status`实物退货；`refund_status`退款状态；`refund_processing_status`处理阶段；`reshipment_status`补发；`compensation_status`补偿；`extra_json`扩展信息 |
| rule_documents | `policy_id`主键；`title`标题；`rule_type`平台/店铺规则；`normalized_rule`摘要；`source_level`来源等级；`review_status`审核；`source_url`来源；`raw_rule_json`完整规则卡 |
| usage_documents | `document_id`主键；`product_id`旧文档商品关联（可以为空）；`title/content/source`标题、正文、来源 |
| usage_document_metadata | `document_id`主键兼外键；`version/effective_time`版本及标记日期；`review_status`审核；`is_simulated`是否模拟；`warnings_json`警告；`raw_document_json`完整文档 |
| usage_document_bindings | `binding_id`自增主键；`document_id/snapshot_id`文档与快照；`product_id/sku_id/product_name`绑定对象；`match_basis`绑定依据；文档＋快照唯一 |
| fact_import_metadata | `source_path`来源文件主键；`sha256`文件校验；`imported_at`UTC同步时间 |

`extra_json`保留如下实际存在的可选字段：商品为 `product_variants/related_product_ids/shelf_life/use_case`；SKU为 `related_sku_ids/promotion/free_shipping/shipping_fee`；订单为 `related_order_ids/merge_shipment_status/delivery_mode`；物流为 `default_carriers/related_tracking_nos/carrier_restriction/carrier_service_area`；售后为 `freight_insurance_status/return_freight_rule/packaging_status`。不在字段中代表源快照没有提供，不补默认事实。

## 4. 所有工具的公共输入输出

Python内部输入有 `conn`、`snapshot_id`；这些由应用注入，模型不得传入。说明书服务的模型、Top-K和语料版本也由应用配置。业务查询可用 `execute_tool(ToolAction(name,args), session_context, conn, manual_service)`；完整tool_calls协议（包括 `finish_collection`）使用 `dispatch_graph_call`，因为证据索引来自图的消息历史。

工具统一返回以下字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| ok | bool | 查询是否完成并取得结果；partial也可能为true |
| tool | str | 工具名 |
| status | str | found / partial / no_match / forbidden / conflict / tool_error |
| data | dict | 业务数据，见下文各工具 |
| missing_fields | list[str] | 工具预定义字段中的缺失项，不是当前问题充分性的最终结论 |
| evidence_refs | list[dict] | 表、快照、订单或文档章节来源 |
| error_code / error_message | str或null | 可机读错误码和诊断信息 |

`null/unknown`不能转成确定事实，`not_applicable`表示不适用。`found`不等于充分，`partial`不等于必须拒答。不同工具的partial字段集合不同，LLM A仍须按顾客问题判断。当前各事实工具将None、空串、unknown标记为缺失；不会统一标准化原始状态字符串。

## 5. 每个工具的准确契约

### get_recent_orders

- 文件：`orders/recent.py`；读取orders并LEFT JOIN products、skus。
- 模型输入：`limit`可选，整数1～20，默认5；`product_hint`可选非空字符串。
- 应用输入：`visible_snapshot_ids`。None只查当前快照；空数组查不到任何订单；列表顺序决定候选顺序。
- `data`：`orders`数组、`total`过滤后候选总数、`has_more`是否被limit截断、`ordering`固定application_scope_not_chronological、`ownership_verified`固定false。
- 每个orders元素：`order_id/product_id/product_name/sku_id/payment_status/shipment_status/order_time`，其中order_time固定null，因为数据源无下单时间。
- 有候选时status为partial，missing_fields为order_time、user_order_ownership。无候选为no_match。product_hint是大小写无关的字面包含匹配，不能理解语义。
- 不自动确定目标；多候选需LLM A消歧。无用户归属，不得把授权演示范围宣称为已验证的真实顾客订单。

### get_order_info

- 文件：`orders/info.py`；读orders；订单成功查询后执行器会从同快照products、skus更新State绑定，但返回data不增加商品字段。
- 模型输入：必填 `order_id`。
- `data`：`order_id/order_record_status/order_item_status/payment_status/shipment_status/address_status/address_change_status/order_change_status`。
- missing_fields检查order_id、order_record_status、payment_status、shipment_status，其他字段未知不影响found标记。
- 匹配不到返回ORDER_NOT_FOUND；来源引用含orders、snapshot_id、order_id。

### get_logistics

- 文件：`shipping/logistics.py`；读logistics。
- 模型输入：必填 `order_id`，无需机械地先调订单详情。
- `data`：`order_id/tracking_no/actual_carrier/logistics_status/estimated_delivery/arrival_eta/dispatch_commitment`。
- missing_fields检查order_id、logistics_status；找不到返回LOGISTICS_NOT_FOUND。
- 不返回逐站物流事件或确定到达保证，缺少这些事实时不能补写。

### get_product_info

- 文件：`products/info.py`；读products和skus，同快照精确匹配。
- 模型输入：`product_id`或`sku_id`至少一个非空；两者都传时必须均匹配。
- `data`：`product_id/product_name/material/spec/size/product_match/stock_status`（有商品行时）；`sku_id/product_id/price_text/purchase_limit`（有SKU行时），SKU库存优先；`product_extra/sku_extra`为解析后的扩展对象。
- missing_fields检查product_id、product_name、sku_id、material、spec、size、price_text、stock_status；不新增缺失事实。没有SKU时某些返回键可能不存在。
- 保存成功结果到State后更新当前商品/SKU，后续可查说明书。当前工具不是全店商品搜索，不能只传自然语言名称。

### get_after_sales

- 文件：`after_sales/status.py`；读after_sales。
- 模型输入：必填非空 `order_id`。
- `data`：`order_id/after_sales_status/after_sales_request_status/return_status/refund_status/refund_processing_status/reshipment_status/compensation_status/extra`。
- 字段None、空串或unknown会进入missing_fields；不适用的状态仍按原文返回。无行返回AFTER_SALES_NOT_FOUND。
- 没有售后单金额或到账凭证，不把processed等同于顾客已收款。

### search_policies

- 文件：`policies/search.py`、`policies/keyword.py`；读rule_documents。
- 模型输入：必填非空 `query`；可选整数 `limit` 1～20，默认5。
- `data`：`results`数组、`retrieval_backend=legacy_keyword_sqlite_cards`。
- 每项字段：`policy_id/title/normalized_rule/source_level/review_status/source_url/version/score/match_reasons`，以及原卡中可选的 `conditions/scope/evidence_required/must_mention/must_not_say/escalation_when/effective_time/temporal_status`。
- 只允许approved_for_simulation、approved_for_rule_context_with_temporal_note；本地为12＋1条，其余6条pending不参与。
- 以triggers/domain/title/normalized_rule计算完整短语奖励（8/4/3/2）和词项交集分；中文额外拆二字片段。保留正分结果，按分数倒序、policy_id稳定排序，返回Top-limit。没有向量索引、没有Cross-Encoder，也没有把所有规则文本自动判定为当前有效政策。
- found只表示有检索命中；条件、例外、时效仍需LLM A结合事实判断。无正分为no_match。

### search_product_manual

- 冻结商品说明：`manuals/search.py`，底层 `rag/direct_cross_encoder.py`；旧类别模拟说明：`manuals/legacy.py`，直接读取原 SQLite。
- 模型输入只允许 `query`，非空字符串。应用必须先绑定snapshot_id、current_product_id、current_sku_id，禁止模型传商品ID、SKU、manual_id或top_k。
- `data`：`snapshot_id/product_id/sku_id/manual_id/query/results/retriever_config`。
- results每项：`rank/section_id/source_chunk_id/section_title/content/score/manual_id/manual_version/chunk_sha256`。
- retriever_config：`retriever_backend/model_name/model_revision/top_k/max_length/device/passage_template/chunks_sha256/manuals_sha256/corpus_sha256`。
- 证据引用包含source、manual_id、section_id、source_chunk_id、chunk_sha256。
- 缺query、商品或SKU返回partial；没有精确冻结绑定返回no_match；推理异常返回tool_error。匹配到排名结果为found，即使正文不包含用户所问答案也可能found；不得用分数冒充拒答阈值。

## 6. 说明书RAG到底有多大、怎么检索

`frozen_v1` 模式读取 `data/processed/product_manuals/frozen_v1/`，**8份说明书、48个章节，每份6章**。全文JSONL约20,845 bytes，章节JSONL约34,846 bytes（精确哈希见资产清单）。只有8组冻结 `snapshot_id＋product_id＋sku_id` 绑定允许查询，不自动共享给其他快照。`legacy_sqlite` 模式读取原 SQLite 中 8 份类别说明和 236 条旧绑定；结果标明 `source_tier=legacy_category` 与 `is_product_specific=false`。

检索过程：精确绑定对象 → 找该商品6个章节 → 组成query/passage对 → BAAI/bge-reranker-v2-m3逐对打分 → 排序 → 默认返回Top-5。Top-K由应用固定为3/5/6之一。这里直接对6段全部评分，没有先建向量库再召回；其他TF-IDF/BGE向量实验在rag目录，不是此工具的默认后端。

Passage拼接商品名、章节标题和原文；模型revision为 `995ec6ee29e8a96b27eee66c584c4340104ab8e5`，默认max_length=512，优先截断passage。原始分数不是概率。服务初始化一次并跨会话复用，启动检查manifest、文件SHA-256及每章内容哈希；冻结语料不允许原地改写。

旧SQLite中的8份说明书＋236条绑定由另一套模拟脚本生成，与这8份冻结说明书不是同一套可互换数据。Agent 仅在明确选择 `legacy_sqlite` 时读取旧表；比较实验时必须注明入口和语料版本。

## 7. 规则RAG资产与当前工具的区别

当前Agent使用19条SQLite规则卡中的13条审核允许记录做关键词检索。另有 `data/processed/official_rules/retrieval_v1/policy_chunks_v1.jsonl`，**6条父规则、24个chunk**，文件约115,079 bytes；当前均为pending_review、retrieval_ready=false，未接入工具。不能把它与13条旧卡的覆盖率合并统计，也不能因为目录名含RAG就视为已建立检索索引。完整原文及规则治理资产仍在official_rules目录。

## 8. 依赖、隔离与并行

- 需要定位订单才调用get_recent_orders；已有订单号直接查详情、物流或售后。
- 商品详情需要ID/SKU；说明书需要应用已绑定完整对象；规则检索可直接用问题开始。
- 选择候选订单后，执行器只在visible_snapshot_ids内匹配，不全库兜底；跨目标时清除旧对象证据。
- 工具参数引用snapshot_id、user_id、visible_snapshot_ids等会被拒绝，不能靠Prompt保护代替执行器检查。
- 当前执行器串行；不要直接用线程共享同一个SQLite连接或可变SessionState。未来并行需要独立连接/执行上下文与明确的结果合并策略。

## 9. AIMessage原生调用与Prompt

Prompt在 `prompts/agent_tool_system_v1.txt`；工具schema在 `agent/tool_calling.py`。绑定7个业务工具，另提供3个**控制信号**：`finish_collection(evidence_indices)`、`request_fallback(reason_code)`、`handoff_to_human(reason_code, reason_detail)`。控制信号不是数据库查询，也不直接调用LLM B或联系人工；图的条件边收到route后负责执行。转人工原因码包括 `user_requested`、`conflicting_facts`、`high_risk_dispute`、`refund_dispute`、`permission_required`、`tool_unavailable`、`other`。

LLM A不输出正文调用JSON；模型通过原生tool-calling产生消息，由LangChain解析成AIMessage.tool_calls，其内部args仍是结构化字典。AIMessage是Python消息对象，不是要求模型逐字生成的文本。模型及服务端必须支持相应工具调用协议，不能仅靠Prompt让普通纯文本SFT自动拥有这一能力。

`agent/tool_calling.py` 现在是唯一的工具目录：`tool_definitions()` 给模型提供JSON schema，`dispatch_graph_call()` 按同一份schema校验并执行LangGraph工具调用。旧 `dispatch_native_call()` 仍可直接执行业务查询，但不能处理 `finish_collection`，以免把 `SessionState.evidence` 的索引误当成消息证据索引。LangGraph `State` 保存 `messages`、`tool_call_count`，以及转人工后的 `service_status` 和 `handoff_result`；数据库范围和当前对象由应用创建的 `SessionState` 作为可信查询上下文传入。示例（`graph_state`、`session_context`、`conn`、`chat_model`由宿主创建）：

```python
from agent.tool_calling import tool_definitions, dispatch_graph_call

planner = chat_model.bind_tools(tool_definitions())
ai_message = planner.invoke(graph_state["messages"])
call = ai_message.tool_calls[0]
observation = dispatch_graph_call(call, graph_state, session_context, conn, manual_service)
# nodes/tools.py 把结果包装成带原call id的ToolMessage并写入图状态。
# 控制工具返回的route仍需由图分支处理。
```

`agent/nodes/tools.py` 中的 `make_tool_node(conn, session_context, manual_service)` 可串行执行工具并返回 `messages` 和 `tool_call_count` 的增量更新。业务工具会修改共享的 `session_context`，因此恢复会话时应用也必须恢复对应的可信查询上下文；只恢复 LangGraph 消息不足以恢复订单绑定。执行器按唯一schema校验模型参数并拒绝 `snapshot_id` 等额外字段；会话范围仍由业务执行器校验。`finish_collection` 从工具消息推导可用结果并检查索引合法性；语义是否充分仍由 LLM A 判断。LangGraph路径需要安装LangChain Core和LangGraph。

`handoff_to_human` 的工具结果只表示转接请求已通过参数校验。将其路由至 `make_handoff_node(client, session_context)` 后，节点调用注入的客服接口；只有接口同时返回 `success=True`、`handoff_status="transferred"` 和非空 `ticket_id`，才将 `service_status` 更新为 `transferred`。接口异常或失败时状态为 `handoff_failed`。本地可用 `SimulatedHandoffClient` 演示，它返回 `SIM-` 开头的工单号，不能当作真实人工客服已接通。

以下是原生schema接入示例（无需LangChain适配器；未运行真实模型）：

```python
import json
from pathlib import Path
from langchain_core.messages import SystemMessage, ToolMessage
from agent.tool_calling import tool_definitions, dispatch_graph_call

prompt = Path("prompts/agent_tool_system_v1.txt").read_text()
planner = chat_model.bind_tools(tool_definitions())  # chat_model由宿主配置
messages = [SystemMessage(content=prompt), *history]
ai_message = planner.invoke(messages)
messages.append(ai_message)
if ai_message.invalid_tool_calls:
    # 调用未成功解析：由宿主记录错误并限制重试，不执行正文文本。
    raise ValueError("invalid native tool call")
if not ai_message.tool_calls:
    # 本Prompt规定正文只用于澄清；展示后保存状态并等待用户补充。
    clarification = ai_message.content
else:
    # 默认每轮一个调用。宿主应在执行前验证数量、参数、预算；
    # 不允许业务调用与终止信号混合后继续盲目执行。
    if len(ai_message.tool_calls) != 1:
        raise ValueError("this runner expects one call per turn")
    call = ai_message.tool_calls[0]
    observation = dispatch_graph_call(call, {"messages": messages}, session_context, conn, manual_service)
    messages.append(ToolMessage(
        content=json.dumps(observation, ensure_ascii=False),
        tool_call_id=call["id"], name=call["name"]))
    # observation.route=respond/fallback/handoff -> 图的对应节点；
    # 普通查询结果或校验错误 -> 返回同一个LLM A节点。
```

LangGraph路径中，宿主每轮需提供可信会话的当前对象；适配器在可用业务工具结果中加入 `evidence_index`，这些结果来自 `messages` 中的 `ToolMessage`，不需要在Graph State里另设 `evidence`。不要将查询上下文中的隐藏范围配置直接喂给模型。finish_collection只检查证据索引合法与来源状态，不替模型判断语义充分性；partial可以支持某些问题。澄清内容不写入SFT最终回复目标。若将来训练LLM A，保存原生assistant tool_calls及逐个对应的tool消息，不能仅保存正文JSON。

当前桥接不是完整动作校验器/鉴权系统：最多次数、重试、意图充分性、终止信号是否适当、输出审核和恢复都由后续图控制。说明书真实模型推理需torch、transformers；默认本地测试使用FakeScorer，只验证隔离与契约。

协议参考：[LangChain官方工具调用说明](https://www.langchain.com/blog/tool-calling-with-langchain)、[ToolMessage示例接口](https://reference.langchain.com/python/langchain-core/utils/function_calling/tool_example_to_messages)。

## 10. 运行、更新与验证

在项目根目录：

```bash
python3 -m unittest discover -s agent/tests -v
python3 scripts/test_simulated_knowledge.py
python3 scripts/inspect_business_tool_assets.py
python3 scripts/search_simulated_knowledge.py rules "物流一直没更新怎么办"
```

仅同步最新训练事实用 `python3 scripts/sync_customer_service_facts.py`；会生成备份并保留规则/说明书。全新建库用build_customer_service_db.py；不要为整理工具而覆盖现有数据库。源码目录变更保持旧import可用；业务事实和冻结语料本次未改动。
