"""Redis session leases. Durable Graph writes need a separate fencing check."""

import asyncio
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass
from time import monotonic

from redis.asyncio import Redis
from redis.exceptions import RedisError

from .redis import RedisUnavailableError

_RELEASE = "if redis.call('GET',KEYS[1])==ARGV[1] then return redis.call('DEL',KEYS[1]) else return 0 end"
_RENEW = "if redis.call('GET',KEYS[1])==ARGV[1] then return redis.call('PEXPIRE',KEYS[1],ARGV[2]) else return 0 end"


class LockTimeoutError(TimeoutError): pass
class LockOwnershipLostError(RuntimeError): pass


@dataclass(frozen=True)
class LockLease:
    key: str
    owner_token: str
    ttl_ms: int


class SessionLock:
    def __init__(self, redis: Redis, prefix: str = "cs"):
        self.redis, self.prefix = redis, prefix

    def key(self, session_id: str) -> str:
        if not session_id: raise ValueError("session_id is required")
        return f"{self.prefix}:lock:session:{session_id}"

    async def acquire(self, session_id: str, *, ttl_ms: int = 30000,
                      wait_timeout: float = 0, poll_interval: float = 0.05) -> LockLease:
        if ttl_ms < 1 or wait_timeout < 0 or poll_interval <= 0: raise ValueError("invalid lock timing")
        lease = LockLease(self.key(session_id), secrets.token_hex(24), ttl_ms)
        deadline = monotonic() + wait_timeout
        while True:
            try:
                if await self.redis.set(lease.key, lease.owner_token, nx=True, px=ttl_ms):
                    return lease
            except RedisError as exc:
                raise RedisUnavailableError("Redis lock acquisition failed") from exc
            remaining = deadline - monotonic()
            if remaining <= 0: raise LockTimeoutError("session lock wait timed out")
            await asyncio.sleep(min(poll_interval, remaining))

    async def assert_owner(self, lease: LockLease) -> None:
        try:
            owner = await self.redis.get(lease.key)
        except RedisError as exc:
            raise RedisUnavailableError("Redis lock ownership check failed") from exc
        if owner != lease.owner_token: raise LockOwnershipLostError("session lock ownership lost")

    async def renew(self, lease: LockLease) -> None:
        try:
            result = await self.redis.eval(_RENEW, 1, lease.key, lease.owner_token, lease.ttl_ms)
        except RedisError as exc:
            raise RedisUnavailableError("Redis lock renewal failed") from exc
        if result != 1: raise LockOwnershipLostError("session lock ownership lost")

    async def release(self, lease: LockLease) -> None:
        try:
            result = await self.redis.eval(_RELEASE, 1, lease.key, lease.owner_token)
        except RedisError as exc:
            raise RedisUnavailableError("Redis lock release failed") from exc
        if result != 1: raise LockOwnershipLostError("session lock ownership lost")

    @asynccontextmanager
    async def hold(self, session_id: str, **kwargs):
        lease = await self.acquire(session_id, **kwargs)
        try:
            yield lease
        finally:
            await self.release(lease)

    # Pass owner_token into a durable request-execution record and compare it
    # inside the same PostgreSQL transaction as any Graph/checkpoint write.
    # assert_owner alone cannot fence a paused worker after lease expiry.
