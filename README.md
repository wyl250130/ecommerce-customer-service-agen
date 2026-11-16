# Customer Service

FastAPI 通过 PostgreSQL 保存业务消息和请求，通过 LangGraph Checkpointer 保存多轮状态；Redis 和 PostgreSQL 执行锁协调同一会话。Agent 的业务工具仍使用项目内只读 SQLite 模拟数据。应用启动不会自动创建业务表。

## 依赖与配置

使用 uv 管理依赖，版本记录在 `pyproject.toml` 和 `uv.lock`。在项目根目录运行：

```bash
uv sync
cp .env.example .env
```

填写 `.env` 中的 `CS_POSTGRES_USER`、`CS_POSTGRES_PASSWORD`、`CS_POSTGRES_DB` 和 `CS_DATABASE_URL`。连接 URL 使用 `postgresql+asyncpg://...` 格式，密码中有特殊字符时需进行 URL 编码。`CS_DATABASE_URL` 不填时，应用仍可启动并响应 `/health`，数据库依赖会返回 503。

连接池参数由 `CS_DB_POOL_SIZE`、`CS_DB_MAX_OVERFLOW`、`CS_DB_POOL_TIMEOUT`、`CS_DB_POOL_RECYCLE` 和 `CS_DB_POOL_PRE_PING` 配置，默认值见 `.env.example`。应用在启动时建立 AsyncEngine 和会话工厂，在关闭时释放连接池；此过程不主动连接数据库。

## PostgreSQL 与迁移

需要本机安装并启动 Docker。Compose 使用命名 Volume 保存数据，端口仅监听 `127.0.0.1`：

```bash
docker compose up -d postgres
docker compose ps
uv run alembic upgrade head
uv run alembic current
```

首次运行数据库集成测试前，创建与 `.env` 中 `CS_TEST_DATABASE_URL` 对应的测试库：

```bash
docker compose exec -T postgres sh -c 'createdb -U "$POSTGRES_USER" customer_service_test'
```

该命令只需执行一次；测试库已存在时无需重复创建。

首个修订 `0001_infrastructure` 只建立 Alembic 版本记录。后续 ORM 模型继承 `backend.app.models.base.Base`，新增模型后需在 `backend/migrations/env.py` 中导入模型模块，再使用 `uv run alembic revision --autogenerate -m "说明"` 生成迁移并审核。迁移使用 `CS_DATABASE_URL`，不在应用启动时自动执行。

## 启动与测试

```bash
uv run uvicorn backend.app.main:app --reload
curl http://127.0.0.1:8000/health
uv run pytest backend/tests -q
```

`/health` 预期返回 `{"status":"ok"}`。数据库集成测试需在 `.env` 中设置 `CS_TEST_DATABASE_URL`，指向可创建和删除测试表的**一次性 PostgreSQL 数据库**。未设置时，这些测试会跳过。测试覆盖连接、请求事务提交与回滚、会话归还连接池，以及 `alembic upgrade head`。不要将测试 URL 指向生产数据库。

## 下一阶段可用接口

- `get_settings()`：读取 `CS_` 应用和数据库配置。
- `create_database_engine(settings)`：创建 SQLAlchemy AsyncEngine。
- `create_session_factory(engine)`：创建 `async_sessionmaker[AsyncSession]`。
- `get_session(request)`：FastAPI `Depends` 依赖；请求成功提交，异常回滚，退出时关闭 Session。
- `app.state.db_engine` 与 `app.state.session_factory`：应用生命周期内共享的引擎和会话工厂。
- `Base.metadata`：后续 ORM 模型与 Alembic 的共享元数据。

现有 Agent 仍使用自己的 SQLite 数据源，本次 PostgreSQL 配置不改变它。

## Redis 协调基础设施

在 `.env` 中配置 `CS_REDIS_URL=redis://127.0.0.1:6379/0`，测试使用独立的 `CS_TEST_REDIS_URL=redis://127.0.0.1:6379/15`。连接数、超时、健康检查间隔和 key 前缀由 `CS_REDIS_*` 配置。启动依赖：

```bash
docker compose up -d postgres redis
docker compose ps
uv run pytest backend/tests -q
```

`GET /health` 只检查进程是否响应；`GET /ready` 查询 PostgreSQL 与 Redis，任一未配置或不可用时返回 503。Redis 客户端在应用 lifespan 中创建并关闭。

`SessionLock` 使用唯一 owner token 和 Redis 原子脚本保护释放、续期；调用方可在执行期间调用 `assert_owner()`。租约到期或 Redis 故障会抛出明确异常。`RateLimiter` 使用原子固定窗口计数；`JsonCache` 仅用于明确允许的非敏感、可重建数据，不存订单、Graph State 或 RAG Evidence。Redis 不是业务事实或 Checkpoint 的持久化来源。

## LangGraph PostgreSQL Checkpointer

设置 `CS_CHECKPOINT_ENABLED=true` 后，应用 lifespan 使用独立的 psycopg 异步连接池创建 `AsyncPostgresSaver`，执行一次幂等的 `setup()`，关闭时释放连接池。连接从 `CS_DATABASE_URL` 转换为 psycopg DSN；连接池使用 autocommit、dict_row 和严格的 msgpack 内置类型白名单，不允许任意自定义类型或 pickle 回退。Checkpoint 内部表由 LangGraph 自行维护，不属于 Alembic 迁移或业务 AsyncSession。

配置 `CS_LLM_API_KEY`、`CS_LLMA_MODEL`、`CS_LLMB_MODEL` 后，应用启动时各创建一个长期复用的 `ChatOpenAI` 模型；`CS_LLM_BASE_URL` 和 `CS_LLM_TIMEOUT` 可选。没有密钥时，应用工厂仍可注入返回 `(llma_model, llmb_model)` 的 `model_factory` 进行测试：

```python
app = create_app(model_factory=my_model_factory)
```

应用内从可信会话记录取得 `thread_id`，首次用 `make_initial_state([HumanMessage(...)], context=context)`；后续用 `{"messages": [HumanMessage(...)]}` 追加消息。两次调用都传入新的可信 `RuntimeContext` 和 `{"configurable": {"thread_id": thread_id}}`。通过 `await app.state.invoke_graph(input_state, config=config, context=context)` 执行；通过 `await app.state.graph.aget_state(config)` 读取保存状态。重启后新 Graph 使用相同 thread ID 可恢复。不要把客户端提交的任意 thread ID 直接用于读取或执行。

`invoke_graph` 先取得 Redis 租约，再在一条专用 psycopg 连接上取得该 thread 的 PostgreSQL session advisory lock。`execution_guards` 表在短事务中记录 `request_id`、`owner_token`、单调 `execution_epoch` 和执行状态。一次运行的 `AsyncPostgresSaver` 只使用持锁的这条连接，并以 `durability="sync"` 执行 Graph；因此 `aput`、`aput_writes` 等由该 Saver 发出的写入不能在持锁连接断开后换连接继续。关闭连接同时终止旧 Saver 的写能力并释放 advisory lock，新 owner 才能取得锁并增加 epoch。LLM 推理期间不占用业务 AsyncSession 事务。

Redis 丢锁时会取消 Graph、等待其任务结束，再关闭 PostgreSQL 连接；若任务不响应取消，新 owner 仍无法取得数据库锁。进程崩溃时数据库会释放连接锁，下一次执行可增加 epoch 并恢复 thread。`app.state.graph` 只提供 `aget_state()`；持久化执行必须走 `app.state.invoke_graph(input_state, config=config, context=context, request_id=request_id)`。`request_id` 应来自应用创建的 ChatRequest，thread ID 应来自已授权会话。直接创建其他 Saver 或直接向 Checkpoint 表写入不受这条执行路径保护；未来真实工具副作用还需单独的幂等和所有权策略。测试使用随机 thread ID，完成后删除对应 Checkpoint。

## 业务持久化结构

`backend/app/models/business.py` 定义八张业务表：`users`（用户身份）、`sessions`（会话、`thread_id`、服务状态）、`messages`（按会话唯一的消息序号）、`chat_requests`（按会话唯一的幂等键和执行结果）、`handoff_requests`（人工接管请求）、`agent_assignments`（人工分配历史）、`audit_logs`（审计记录）、`outbox_events`（待投递事件与重试状态）。人工分配使用 `released_at IS NULL` 的部分唯一索引，保证每个会话最多一个有效分配。时间列使用带时区类型，应用生成的时间使用 UTC。

正式表结构由 `backend/migrations/versions/587227d01c1d_business_schema.py` 迁移创建；应用启动不会自动建表。Repository 位于 `backend/app/repositories/business.py`，只使用调用方传入的 `AsyncSession`，写入后 `flush`，由外层事务决定提交或回滚。列表查询使用稳定排序与游标；需要竞争控制的状态变化使用条件更新。

Application Service 后续可这样组合 Repository（示例省略请求参数校验）：

```python
from backend.app.repositories.business import UserRepository, SessionRepository

async with app.state.session_factory.begin() as db:
    user = await UserRepository(db).create(external_id)
    conversation = await SessionRepository(db).create(user.id, thread_id)
# 离开 begin 块时统一提交；异常时统一回滚并关闭 Session。
```

HTTP 路由可使用 `Depends(get_session)` 获取同样的请求级事务。Repository 的 `DuplicateKeyError` 表示数据库唯一约束冲突；状态转换返回 `False` 表示预期状态已改变，调用方应重新读取并决定后续动作。业务表不保存 LangGraph 内部的 ToolMessage、Evidence 或 Checkpointer 表。

## HTTP 聊天链路

运行前执行 `uv run alembic upgrade head`，将 `CS_CHECKPOINT_ENABLED=true`，填写真实模型配置和 `CS_DEFAULT_SNAPSHOT_ID`。该 snapshot ID 必须来自服务器侧 SQLite 模拟数据的 `orders.snapshot_id`。本地开发可设置 `CS_DEV_USER_EXTERNAL_ID` 为固定身份；该身份不接受客户端传入的 user_id、thread_id 或 snapshot_id。`CS_ENVIRONMENT` 为其他值时，此开发身份入口会拒绝请求，需接入正式认证后再对外开放。

```bash
docker compose up -d postgres redis
uv run alembic upgrade head
uv run uvicorn backend.app.main:app --reload
curl -X POST http://127.0.0.1:8000/api/sessions
curl -X POST http://127.0.0.1:8000/api/sessions/SESSION_ID/messages \
  -H 'Content-Type: application/json' \
  -d '{"content":"查询我的订单","idempotency_key":"local-turn-1"}'
curl http://127.0.0.1:8000/api/sessions/SESSION_ID/messages
curl http://127.0.0.1:8000/api/chat-requests/REQUEST_ID
```

POST 消息接口返回请求 ID、执行状态、`result_type`（`completed`、`needs_user`、`fallback` 或 `handoff`）和最终回复。每次请求先短事务保存用户消息及幂等键，再从可信会话读取 thread ID、构造 RuntimeContext 并调用 `app.state.invoke_graph()`；Graph 完成后用另一个短事务保存助手消息和请求状态。相同幂等键不会重复运行 Agent。若 Checkpoint 已完成而最终业务写入失败，相同请求重试会在取得执行锁后从 Checkpoint 补写结果。用户可见消息通过 `request_id` 去重，顺序由会话内序号确定。

测试命令：`uv run pytest backend/tests -q` 和 `uv run python -m unittest discover -s agent/tests -q`。HTTP 集成测试使用脚本模型及独立测试数据库、Redis 前缀；它验证了真实工具调用与持久化流程，但不代替真实模型服务验收。没有 API Key 时，聊天接口返回 503，`/health` 仍可使用。

### 多轮聊天验收数据

`scripts/build_chat_e2e_facts.py` 从项目现有样本事实库选取一条已明确关联订单、商品、SKU 和物流的模拟快照，构建独立的 `data/processed/customer_service/customer_service_chat_e2e.sqlite`。原始 `customer_service_train.sqlite` 不会被修改。派生快照 `CHAT-E2E-SNAPSHOT-001` 中，唯一新增的事实是用于演示追问的 `logistics.estimated_delivery`；`chat_fixture_metadata` 记录原库 SHA-256、来源快照和该模拟字段。订单号、商品及其他状态均来自现有样本，不跨快照拼接。

```bash
uv run python scripts/build_chat_e2e_facts.py
```

本地多轮验收时，在 `.env` 中将 `CS_FACTS_DATABASE_PATH` 指向新库的绝对路径，并设置 `CS_DEFAULT_SNAPSHOT_ID=CHAT-E2E-SNAPSHOT-001`。两项配置均由服务端读取，客户端不能指定。新库仍是研究用模拟事实，没有可核验的真实用户归属；只有 `CS_ENVIRONMENT=local/test` 的开发身份可使用。需要测试缺失商品或预计送达的兜底行为时，仍使用原始样本库及对应快照。

## 浏览器客服界面

`frontend/` 是 React + TypeScript 应用。先按上文配置 `.env` 并启动 PostgreSQL、Redis，执行迁移。开发时分别启动后端和前端：

```bash
# 终端一，项目根目录
docker compose up -d postgres redis
uv run alembic upgrade head
uv run uvicorn backend.app.main:app --reload

# 终端二，项目根目录；需要先安装 Node.js 和 npm
cd frontend
npm ci
npm run dev
```

浏览器打开 `http://127.0.0.1:5173/`。Vite 将 `/api` 请求转发给本机 8000 端口。构建后可只运行后端，由 FastAPI 在根路径提供页面：

```bash
cd frontend && npm ci && npm run build
cd .. && uv run uvicorn backend.app.main:app --reload
```

此时打开 `http://127.0.0.1:8000/`。前端通过 `GET /api/sessions` 列出当前身份的最近 100 个会话，通过 `GET /api/sessions/{session_id}/chat-requests/by-key/{idempotency_key}` 查找原请求；已有的创建会话、发送消息、历史消息和按 ID 查询请求接口保持不变。刷新页面会从服务端重新读取会话及消息。发送中断后，页面保留原幂等键，先查询请求状态；点击“安全重试”也沿用该键，不会生成新的请求。

当前界面和 HTTP 接口仅提供本地开发身份，页面明确显示“开发模式 · 模拟身份”。尚未接入正式用户认证，不能部署为公开客服服务。模型密钥、数据库凭证、授权快照和 thread ID 只在后端配置与使用；浏览器不传这些字段。`transferred` 状态仅表示模拟转人工，尚无真人工作台。

## 说明书数据与 RAG 对比

`scripts/build_manual_corpus_v2.py` 将旧 SQLite 的 8 份类别说明、236 条绑定和冻结 v1 的 8 份商品说明导出为独立的[统一检索实验集](data/processed/product_manuals/unified_v2/README.md)。导出包含 16 份说明、104 个章节、245 条精确绑定、334 张非说明书的商品事实卡，以及原有 800 条人工编写的冻结说明书检索题。它不修改原库、冻结语料或 Gold。当前 HTTP Agent 尚未切换到此实验集；离线基线和数据边界见该目录 README。

要验证订单、商品、说明书和回复是否来自同一个模拟场景，使用[Agent 全链路事实场景](data/processed/agent_e2e_v1/README.md)。其中 51 条 train 场景加 1 条聊天演示场景用于旧类别说明的数据一致性检查；另有一个隔离 SQLite 副本加入明确标记的模拟订单，现有 Graph 使用脚本模型依次调用订单、商品和冻结说明书工具并生成可核对的回复。原始事实库和冻结说明书不受修改。真实 LLM 的说明书 HTTP 链路尚未验收。

52 条场景的旧类别说明已可选接入 Agent 内部 `search_product_manual` 工具：设置 `CS_ENABLE_MANUAL_SEARCH=true`、`CS_MANUAL_SOURCE=legacy_sqlite`，并让 `CS_FACTS_DATABASE_PATH` 指向含该场景的原 SQLite。该路径直接读取 SQLite 的已批准模拟说明和精确商品/SKU绑定，不读取上面的 RAG 实验集。单个 HTTP 会话当前仍只授权服务端默认快照；按用户绑定多笔候选订单尚未实现。脚本模型已验证从订单、商品到章节及回复的 Graph 路径，真实模型 HTTP 路径尚未验收。

## 公开仓库的数据范围

仓库只包含运行与验证所需的模拟 SQLite 事实库、模拟商品说明书和检索评测样本。原始训练 Gold、个人或企业数据、本地 `.env`、缓存与构建产物不上传。`.env.example` 只列配置项，不含凭证。`agent/tests/test_business_tools.py` 的训练 Gold 来源一致性检查会在该文件缺席时跳过；其他业务工具测试仍运行。
