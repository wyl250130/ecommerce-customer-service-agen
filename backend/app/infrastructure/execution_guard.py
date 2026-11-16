"""Single-writer PostgreSQL connection and durable epoch for each Graph run.

Every checkpoint write uses the same psycopg connection that holds the advisory
lock. If that connection dies, the saver cannot reconnect or write stale state.
"""

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from time import monotonic

from psycopg import AsyncConnection
from psycopg.rows import dict_row

from .checkpoint import checkpoint_dsn


class ExecutionBusyError(RuntimeError): pass
class ExecutionOwnershipLostError(RuntimeError): pass


@dataclass(frozen=True)
class ExecutionOwner:
    thread_id: str
    request_id: str
    owner_token: str
    execution_epoch: int


class ExecutionGuard:
    def __init__(self, conn: AsyncConnection, owner: ExecutionOwner):
        self.conn, self.owner = conn, owner

    @classmethod
    async def acquire(cls, database_url: str, thread_id: str, request_id: str,
                      owner_token: str, wait_timeout: float = 0) -> "ExecutionGuard":
        if not all((thread_id, request_id, owner_token)):
            raise ValueError("thread_id, request_id and owner_token are required")
        conn = await AsyncConnection.connect(checkpoint_dsn(database_url),
            autocommit=True, row_factory=dict_row, prepare_threshold=0)
        try:
            deadline = monotonic() + wait_timeout
            while True:
                row = await (await conn.execute(
                    "SELECT pg_try_advisory_lock(hashtextextended(%s, 0)) AS acquired",
                    (thread_id,))).fetchone()
                if row["acquired"]: break
                if monotonic() >= deadline: raise ExecutionBusyError("thread execution already owned")
                await asyncio.sleep(min(0.05, max(0, deadline - monotonic())))
            async with conn.transaction():
                row = await (await conn.execute(
                    "SELECT execution_epoch FROM execution_guards WHERE thread_id=%s FOR UPDATE",
                    (thread_id,))).fetchone()
                if row is None:
                    epoch = 1
                    await conn.execute("""INSERT INTO execution_guards
                        (thread_id,execution_epoch,request_id,owner_token,status)
                        VALUES (%s,%s,%s,%s,'running')""",
                        (thread_id, epoch, request_id, owner_token))
                else:
                    epoch = row["execution_epoch"] + 1
                    await conn.execute("""UPDATE execution_guards SET execution_epoch=%s,
                        request_id=%s,owner_token=%s,status='running',updated_at=now()
                        WHERE thread_id=%s""", (epoch, request_id, owner_token, thread_id))
            return cls(conn, ExecutionOwner(thread_id, request_id, owner_token, epoch))
        except BaseException:
            await conn.close()
            raise

    async def _transition(self, status: str) -> None:
        owner = self.owner
        result = await self.conn.execute("""UPDATE execution_guards SET status=%s,updated_at=now()
            WHERE thread_id=%s AND execution_epoch=%s AND request_id=%s
            AND owner_token=%s AND status='running'""",
            (status, owner.thread_id, owner.execution_epoch, owner.request_id, owner.owner_token))
        if result.rowcount != 1: raise ExecutionOwnershipLostError("execution ownership lost")

    async def assert_owner(self) -> None:
        owner = self.owner
        row = await (await self.conn.execute("""SELECT 1 FROM execution_guards
            WHERE thread_id=%s AND execution_epoch=%s AND request_id=%s
            AND owner_token=%s AND status='running'""",
            (owner.thread_id, owner.execution_epoch, owner.request_id, owner.owner_token))).fetchone()
        if row is None: raise ExecutionOwnershipLostError("execution ownership lost")

    async def renew(self) -> None:
        owner = self.owner
        result = await self.conn.execute("""UPDATE execution_guards SET updated_at=now()
            WHERE thread_id=%s AND execution_epoch=%s AND request_id=%s
            AND owner_token=%s AND status='running'""",
            (owner.thread_id, owner.execution_epoch, owner.request_id, owner.owner_token))
        if result.rowcount != 1: raise ExecutionOwnershipLostError("execution ownership lost")

    async def complete(self) -> None: await self._transition("completed")
    async def fail(self) -> None: await self._transition("failed")

    async def close(self) -> None:
        # Closing the connection makes future saver writes impossible and
        # releases its session lock as one server-side connection teardown.
        await self.conn.close()
