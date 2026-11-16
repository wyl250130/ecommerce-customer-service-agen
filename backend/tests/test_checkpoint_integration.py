import asyncio
import sqlite3
from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from psycopg_pool import PoolTimeout

from agent.context import RuntimeContext
from agent.state import make_initial_state
from agent.tools.common import configure_connection
from backend.app.infrastructure.app_config import AppSettings, get_settings
from backend.app.infrastructure.checkpoint import checkpoint_dsn, invoke_serialized, open_checkpointer
from backend.app.infrastructure.locks import LockOwnershipLostError, LockTimeoutError, SessionLock
from backend.app.infrastructure.redis import create_redis_client, close_redis
from backend.app.main import create_app


ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data/processed/customer_service/customer_service_train.sqlite"


class ScriptedModel:
    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return AIMessage(content="模拟回复")


def _models():
    return ScriptedModel(), ScriptedModel()


def test_checkpoint_persistence_and_isolation(monkeypatch):
    url = AppSettings().test_database_url
    if not url:
        pytest.skip("CS_TEST_DATABASE_URL is required")
    monkeypatch.setenv("CS_DATABASE_URL", url)
    monkeypatch.setenv("CS_CHECKPOINT_ENABLED", "true")
    get_settings.cache_clear()
    thread_a, thread_b = f"test-{uuid4()}", f"test-{uuid4()}"

    async def run():
        with configure_connection(sqlite3.connect(DB)) as conn:
            snapshot = conn.execute("SELECT snapshot_id FROM orders LIMIT 1").fetchone()[0]

        def context():
            return RuntimeContext(
                connection_factory=lambda: configure_connection(sqlite3.connect(DB)),
                default_snapshot_id=snapshot, visible_snapshot_ids=(snapshot,),
            )

        config_a = {"configurable": {"thread_id": thread_a}}
        config_b = {"configurable": {"thread_id": thread_b}}
        app1 = create_app(model_factory=_models)
        try:
            async with app1.router.lifespan_context(app1):
                graph = app1.state.graph
                assert graph is not None
                initial = make_initial_state([HumanMessage(content="第一轮")], context=context())
                initial.update(current_order_id="ORDER-SAVED", current_product_id="PRODUCT-SAVED",
                               evidence_generation=3)
                first = await app1.state.invoke_graph(initial,
                    config=config_a, context=context())
                assert first["messages"][-1].content == "模拟回复"
                saved = await graph.aget_state(config_a)
                assert saved.values["tool_call_count"] == 0
                second = await app1.state.invoke_graph({"messages": [HumanMessage(content="第二轮")]},
                                             config=config_a, context=context())
                assert second["current_order_id"] == "ORDER-SAVED"
                assert second["current_product_id"] == "PRODUCT-SAVED"
                assert second["evidence_generation"] == 3
                assert sum(isinstance(m, HumanMessage) for m in second["messages"]) == 2
                await app1.state.invoke_graph(make_initial_state(
                    [HumanMessage(content="独立会话")], context=context()),
                    config=config_b, context=context())
                assert (await graph.aget_state(config_b)).values["current_order_id"] is None
            assert app1.state.checkpointer is not None
            assert app1.state.checkpointer.conn.closed

            app2 = create_app(model_factory=_models)
            async with app2.router.lifespan_context(app2):
                restored = await app2.state.graph.aget_state(config_a)
                assert restored.values["current_order_id"] == "ORDER-SAVED"
                assert restored.values["current_product_id"] == "PRODUCT-SAVED"
                assert restored.values["evidence_generation"] == 3
                third = await app2.state.invoke_graph(
                    {"messages": [HumanMessage(content="重启后第三轮")]},
                    config=config_a, context=context())
                assert sum(isinstance(m, HumanMessage) for m in third["messages"]) == 3
                await app2.state.checkpointer.adelete_thread(thread_a)
                await app2.state.checkpointer.adelete_thread(thread_b)
        finally:
            get_settings.cache_clear()

    asyncio.run(run())


def test_checkpoint_url_and_database_failure():
    assert checkpoint_dsn("postgresql+asyncpg://user:p%40ss@localhost/db") == \
        "postgresql://user:p%40ss@localhost/db"

    async def run():
        settings = AppSettings(_env_file=None,
            database_url="postgresql+asyncpg://127.0.0.1:1/unavailable",
            checkpoint_pool_min_size=1, checkpoint_pool_timeout=0.3)
        with pytest.raises(PoolTimeout):
            await open_checkpointer(settings)

    asyncio.run(run())


def test_thread_execution_requires_live_redis_lease():
    url = AppSettings().test_redis_url
    if not url:
        pytest.skip("CS_TEST_REDIS_URL is required")

    class SlowGraph:
        async def ainvoke(self, state, *, config, context):
            await asyncio.sleep(0.3)
            return state

    async def run():
        client = create_redis_client(AppSettings(_env_file=None, redis_url=url))
        thread = f"test-{uuid4()}"
        lock = SessionLock(client, f"test:{uuid4()}")
        config = {"configurable": {"thread_id": thread}}
        try:
            first = asyncio.create_task(invoke_serialized(SlowGraph(), {}, config=config,
                context=None, lock=lock, ttl_ms=1000, wait_timeout=0))
            await asyncio.sleep(0.03)
            with pytest.raises(LockTimeoutError):
                await invoke_serialized(SlowGraph(), {}, config=config,
                    context=None, lock=lock, ttl_ms=1000, wait_timeout=0.03)
            await first

            lost = asyncio.create_task(invoke_serialized(SlowGraph(), {}, config=config,
                context=None, lock=lock, ttl_ms=120))
            await asyncio.sleep(0.03)
            await client.delete(lock.key(thread))
            with pytest.raises(LockOwnershipLostError):
                await lost
        finally:
            await client.delete(lock.key(thread))
            await close_redis(client)

    asyncio.run(run())
