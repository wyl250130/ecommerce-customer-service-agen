"""Independent psycopg pool for LangGraph checkpoint tables."""

from dataclasses import dataclass
import asyncio
from contextlib import suppress
from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from sqlalchemy.engine import make_url

from .app_config import AppSettings
from .locks import SessionLock


def checkpoint_dsn(sqlalchemy_url: str) -> str:
    """Remove SQLAlchemy's asyncpg driver while retaining encoded credentials."""
    url = make_url(sqlalchemy_url)
    if url.drivername != "postgresql+asyncpg":
        raise ValueError("Checkpoint requires a postgresql+asyncpg application URL")
    return url.set(drivername="postgresql").render_as_string(hide_password=False)


@dataclass
class CheckpointResources:
    pool: AsyncConnectionPool
    saver: AsyncPostgresSaver

    async def close(self) -> None:
        await self.pool.close()


class GraphStateReader:
    """Expose saved state without exposing an unguarded Graph write entry point."""

    def __init__(self, graph: Any):
        self._graph = graph

    async def aget_state(self, config: dict):
        return await self._graph.aget_state(config)


async def open_checkpointer(settings: AppSettings) -> CheckpointResources:
    if not settings.database_url:
        raise ValueError("CS_DATABASE_URL is required for checkpoints")
    if settings.checkpoint_pool_min_size > settings.checkpoint_pool_max_size:
        raise ValueError("checkpoint pool min_size exceeds max_size")
    pool = AsyncConnectionPool(
        conninfo=checkpoint_dsn(settings.database_url),
        min_size=settings.checkpoint_pool_min_size,
        max_size=settings.checkpoint_pool_max_size,
        timeout=settings.checkpoint_pool_timeout,
        reconnect_timeout=settings.checkpoint_pool_timeout,
        kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": 0},
        open=False,
    )
    try:
        await pool.open()
        await pool.wait(timeout=settings.checkpoint_pool_timeout)
        # Strict built-in allowlist. Never deserialize arbitrary custom Python types.
        serde = JsonPlusSerializer(allowed_msgpack_modules=None, pickle_fallback=False)
        saver = AsyncPostgresSaver(pool, serde=serde)
        await saver.setup()
        return CheckpointResources(pool, saver)
    except BaseException:
        await pool.close()
        raise


async def invoke_serialized(graph: Any, input_state: dict, *, config: dict,
                            context: Any, lock: SessionLock, ttl_ms: int = 30000,
                            wait_timeout: float = 5.0) -> dict:
    """Legacy Redis-only coordinator; do not use for checkpointed writes.

    The caller must use an application-authorized thread_id, never one supplied
    directly by an HTTP client. Lease loss cancels the invocation and raises.
    """
    thread_id = config.get("configurable", {}).get("thread_id")
    if not isinstance(thread_id, str) or not thread_id:
        raise ValueError("a trusted thread_id is required")
    async with lock.hold(thread_id, ttl_ms=ttl_ms, wait_timeout=wait_timeout) as lease:
        execution = asyncio.create_task(graph.ainvoke(input_state, config=config, context=context))

        async def heartbeat():
            while True:
                await asyncio.sleep(ttl_ms / 3000)
                await lock.renew(lease)

        renewal = asyncio.create_task(heartbeat())
        try:
            done, _ = await asyncio.wait({execution, renewal}, return_when=asyncio.FIRST_COMPLETED)
            if renewal in done:
                execution.cancel()
                with suppress(asyncio.CancelledError):
                    await execution
                await renewal  # propagate ownership loss or Redis outage
                raise RuntimeError("checkpoint lock renewal stopped")
            result = await execution
            await lock.assert_owner(lease)
            return result
        finally:
            if not execution.done():
                execution.cancel()
                with suppress(asyncio.CancelledError):
                    await execution
            renewal.cancel()
            with suppress(asyncio.CancelledError):
                await renewal
