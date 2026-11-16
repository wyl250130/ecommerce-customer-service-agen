import asyncio
import sqlite3
import time
from pathlib import Path
from uuid import uuid4

import pytest
from psycopg import InterfaceError, OperationalError
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import AIMessage, HumanMessage

from agent.context import RuntimeContext
from agent.state import make_initial_state
from agent.tools.common import configure_connection
from backend.app.infrastructure.app_config import AppSettings
from backend.app.infrastructure.checkpoint import open_checkpointer
from backend.app.infrastructure.execution_guard import (ExecutionBusyError,
    ExecutionGuard, ExecutionOwnershipLostError)
from backend.app.infrastructure.execution_runner import run_owned_graph
from backend.app.infrastructure.locks import LockOwnershipLostError, SessionLock
from backend.app.infrastructure.redis import close_redis, create_redis_client


def test_real_postgres_execution_ownership():
    url = AppSettings().test_database_url
    if not url:
        pytest.skip("CS_TEST_DATABASE_URL is required")

    async def run():
        thread = f"test-{uuid4()}"
        other_thread = f"test-{uuid4()}"
        first = await ExecutionGuard.acquire(url, thread, "request-1", "owner-1")
        try:
            assert first.owner.execution_epoch == 1

            async def contender():
                with pytest.raises(ExecutionBusyError):
                    await ExecutionGuard.acquire(url, thread, "request-2", "owner-2", wait_timeout=0.1)

            async def independent():
                other = await ExecutionGuard.acquire(url, other_thread, "other", "other-owner")
                try:
                    assert other.owner.execution_epoch == 1
                    await other.complete()
                finally:
                    await other.close()

            await asyncio.gather(contender(), independent())
            await first.renew()
            await first.assert_owner()
            await first.complete()
            with pytest.raises(ExecutionOwnershipLostError):
                await first.complete()
        finally:
            await first.close()

        second = await ExecutionGuard.acquire(url, thread, "request-2", "owner-2")
        try:
            assert second.owner.execution_epoch == 2
            stale = ExecutionGuard(second.conn, first.owner)
            with pytest.raises(ExecutionOwnershipLostError):
                await stale.complete()
            with pytest.raises(ExecutionOwnershipLostError):
                await stale.renew()
            await second.fail()
        finally:
            await second.close()

        # A crashed process releases its PostgreSQL session lock. Its old
        # connection cannot write checkpoints after the next owner starts.
        crashed = await ExecutionGuard.acquire(url, thread, "request-3", "owner-3")
        assert crashed.owner.execution_epoch == 3
        stale_saver = AsyncPostgresSaver(crashed.conn)
        await crashed.conn.close()
        recovered = await ExecutionGuard.acquire(url, thread, "request-4", "owner-4")
        try:
            assert recovered.owner.execution_epoch == 4
            with pytest.raises(OperationalError):
                await crashed.complete()
            with pytest.raises((OperationalError, InterfaceError)):
                await stale_saver.aput_writes(
                    {"configurable": {"thread_id": thread, "checkpoint_ns": "",
                                      "checkpoint_id": "old-checkpoint"}},
                    [("old", {"stale": True})], "old-task")
            await recovered.complete()
        finally:
            await recovered.close()

    asyncio.run(run())


def test_redis_loss_cannot_release_postgres_writer_early():
    settings = AppSettings()
    if not settings.test_database_url or not settings.test_redis_url:
        pytest.skip("test PostgreSQL and Redis URLs are required")

    class SlowModel:
        def bind_tools(self, tools): return self
        def invoke(self, messages):
            time.sleep(0.5)
            return AIMessage(content="done")

    async def run():
        db = Path(__file__).resolve().parents[2] / "data/processed/customer_service/customer_service_train.sqlite"
        with configure_connection(sqlite3.connect(db)) as conn:
            snapshot = conn.execute("SELECT snapshot_id FROM orders LIMIT 1").fetchone()[0]
        context = RuntimeContext(connection_factory=lambda: configure_connection(sqlite3.connect(db)),
                                 default_snapshot_id=snapshot)
        resources = await open_checkpointer(AppSettings(_env_file=None,
            database_url=settings.test_database_url))
        redis = create_redis_client(AppSettings(_env_file=None, redis_url=settings.test_redis_url))
        thread = f"test-{uuid4()}"
        lock = SessionLock(redis, f"test:{uuid4()}")
        config = {"configurable": {"thread_id": thread}}
        try:
            task = asyncio.create_task(run_owned_graph((SlowModel(), SlowModel()),
                resources.saver.serde, settings.test_database_url, lock,
                make_initial_state([HumanMessage(content="start")], context=context),
                config=config, context=context, ttl_ms=600))
            await asyncio.sleep(0.05)
            await redis.delete(lock.key(thread))
            with pytest.raises(ExecutionBusyError):
                await ExecutionGuard.acquire(settings.test_database_url, thread,
                    "new-request", "new-owner", wait_timeout=0.05)
            with pytest.raises(LockOwnershipLostError):
                await task
            recovered = await ExecutionGuard.acquire(settings.test_database_url, thread,
                "new-request", "new-owner", wait_timeout=1.0)
            try:
                assert recovered.owner.execution_epoch == 2
                await recovered.complete()
            finally:
                await recovered.close()
        finally:
            await resources.saver.adelete_thread(thread)
            await resources.close()
            await redis.delete(lock.key(thread))
            await close_redis(redis)

    asyncio.run(run())
