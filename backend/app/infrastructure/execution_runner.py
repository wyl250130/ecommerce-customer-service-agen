"""The only application entry point for mutating a checkpointed Graph thread."""

import asyncio
from contextlib import suppress
from uuid import uuid4

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from agent.graph import build_agent
from .execution_guard import ExecutionGuard
from .locks import SessionLock


async def run_owned_graph(models, serde, database_url, lock: SessionLock,
                          input_state, *, config, context, request_id: str | None = None,
                          ttl_ms: int = 30000, wait_timeout: float = 5.0):
    thread_id = config.get("configurable", {}).get("thread_id")
    if not isinstance(thread_id, str) or not thread_id:
        raise ValueError("trusted thread_id is required")
    request_id = request_id or str(uuid4())
    async with lock.hold(thread_id, ttl_ms=ttl_ms, wait_timeout=wait_timeout) as lease:
        guard = await ExecutionGuard.acquire(database_url, thread_id, request_id,
                                             lease.owner_token, wait_timeout=wait_timeout)
        task = None
        renewal = None
        try:
            # The saver cannot reconnect: it writes through the exact connection
            # that owns the PostgreSQL advisory session lock.
            saver = AsyncPostgresSaver(guard.conn, serde=serde)
            graph = build_agent(*models, checkpointer=saver)
            task = asyncio.create_task(graph.ainvoke(input_state, config=config,
                                                     context=context, durability="sync"))

            async def heartbeat():
                while True:
                    await asyncio.sleep(ttl_ms / 3000)
                    await lock.renew(lease)
                    await guard.renew()

            renewal = asyncio.create_task(heartbeat())
            done, _ = await asyncio.wait({task, renewal}, return_when=asyncio.FIRST_COMPLETED)
            if renewal in done:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task  # hold the DB lock until cancellation actually finishes
                await renewal
                raise RuntimeError("execution heartbeat stopped")
            result = await task
            await lock.assert_owner(lease)
            await guard.assert_owner()
            await guard.complete()
            return result
        except BaseException:
            if task is not None and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            with suppress(Exception):
                await guard.fail()
            raise
        finally:
            try:
                if renewal is not None:
                    renewal.cancel()
                    # The failure was already propagated above. Never let a
                    # second await skip releasing the PostgreSQL writer lock.
                    with suppress(asyncio.CancelledError, Exception):
                        await renewal
            finally:
                await guard.close()
