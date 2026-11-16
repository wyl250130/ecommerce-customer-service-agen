# Migrations

Alembic 使用 `env.py` 中的异步 PostgreSQL 连接，并读取 `CS_DATABASE_URL`。
`0001_infrastructure` 只建立 Alembic 版本记录，不创建业务表。后续 ORM 模型继承
`backend.app.models.base.Base`，并在迁移环境中导入模型模块后生成修订。
`587227d01c1d_business_schema` 创建八张应用业务表及约束；迁移环境通过
`backend.app.models` 加载当前模型元数据。
`681eda53217c_execution_guard` 创建每个 thread 的执行所有权记录；
`95c87f0b4a22_chat_request_messages` 为消息增加请求关联及唯一约束，
并为聊天请求保存原始输入和结果类型；
LangGraph 的 `checkpoint_*` 和 `checkpoints` 内部表仍由 Saver 自己管理。
