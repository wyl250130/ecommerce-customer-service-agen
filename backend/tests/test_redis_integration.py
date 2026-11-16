import asyncio
import uuid

import pytest
from fastapi.testclient import TestClient
from redis.asyncio import Redis

from backend.app.infrastructure.app_config import AppSettings, get_settings
from backend.app.infrastructure.cache import JsonCache
from backend.app.infrastructure.locks import (LockLease, LockOwnershipLostError,
    LockTimeoutError, SessionLock)
from backend.app.infrastructure.rate_limiter import RateLimiter
from backend.app.infrastructure.redis import (RedisUnavailableError, check_redis,
    close_redis, create_redis_client)
from backend.app.main import create_app


def test_redis_primitives():
    url = AppSettings().test_redis_url
    if not url:
        pytest.skip("CS_TEST_REDIS_URL is required")

    async def run():
        prefix = f"test:{uuid.uuid4().hex}"
        client = create_redis_client(AppSettings(_env_file=None, redis_url=url))
        try:
            assert await check_redis(client)
            locks = SessionLock(client, prefix)
            async with locks.hold("same", ttl_ms=3000) as lease:
                with pytest.raises(LockTimeoutError):
                    await locks.acquire("same", ttl_ms=3000, wait_timeout=0.05)
                forged = LockLease(lease.key, "wrong-owner", lease.ttl_ms)
                with pytest.raises(LockOwnershipLostError): await locks.release(forged)
                with pytest.raises(LockOwnershipLostError): await locks.renew(forged)
                await locks.assert_owner(lease)
                await locks.renew(lease)
                other = await locks.acquire("different", ttl_ms=500)
                await locks.release(other)
            expired = await locks.acquire("expiry", ttl_ms=80)
            await asyncio.sleep(0.11)
            with pytest.raises(LockOwnershipLostError): await locks.renew(expired)
            replacement = await locks.acquire("expiry", ttl_ms=500)
            with pytest.raises(LockOwnershipLostError): await locks.release(expired)
            await locks.release(replacement)
            extending = await locks.acquire("renewal", ttl_ms=180)
            await asyncio.sleep(0.12)
            await locks.renew(extending)
            await asyncio.sleep(0.12)
            await locks.assert_owner(extending)
            await locks.release(extending)

            limiter = RateLimiter(client, prefix)
            # Fixed windows use Redis server time. Keep this concurrent batch
            # away from a boundary so a legitimate reset cannot split it.
            seconds, micros = await client.time()
            remaining_ms = 2000 - (seconds * 1000 + micros // 1000) % 2000
            if remaining_ms < 500:
                await asyncio.sleep((remaining_ms + 50) / 1000)
            results = await asyncio.gather(*(limiter.check("user", "chat", limit=3, window_ms=2000)
                                             for _ in range(12)))
            assert sum(result.allowed for result in results) == 3
            assert any(result.retry_after_ms > 0 for result in results if not result.allowed)
            await asyncio.sleep(2.05)
            assert (await limiter.check("user", "chat", limit=3, window_ms=2000)).allowed

            cache = JsonCache(client, prefix)
            await cache.set("public", {"value": [1, 2]}, ttl_seconds=1)
            assert await cache.get("public") == {"value": [1, 2]}
            assert await cache.delete("public")
            assert await cache.get("public") is None
            await cache.set("short", 1, ttl_seconds=1)
            await asyncio.sleep(1.05)
            assert await cache.get("short") is None
        finally:
            keys = [key async for key in client.scan_iter(match=f"{prefix}:*")]
            if keys: await client.delete(*keys)
            await close_redis(client)

        bad = Redis.from_url("redis://127.0.0.1:1/15", socket_connect_timeout=0.1,
                             socket_timeout=0.1, decode_responses=True)
        try:
            with pytest.raises(RedisUnavailableError): await SessionLock(bad, prefix).acquire("down")
            with pytest.raises(RedisUnavailableError): await check_redis(bad)
        finally:
            await close_redis(bad)

    asyncio.run(run())


def test_health_and_ready_semantics(monkeypatch):
    settings = AppSettings()
    if not settings.database_url or not settings.redis_url:
        pytest.skip("CS_DATABASE_URL and CS_REDIS_URL are required")
    get_settings.cache_clear()
    try:
        app = create_app()
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
            assert client.get("/ready").json() == {"status": "ready"}
            database = app.state.db_engine
            app.state.db_engine = None
            assert client.get("/ready").status_code == 503
            assert client.get("/health").status_code == 200
            app.state.db_engine = database
            healthy = app.state.redis
            app.state.redis = Redis.from_url("redis://127.0.0.1:1/15",
                socket_connect_timeout=0.1, socket_timeout=0.1)
            try:
                assert client.get("/health").status_code == 200
                assert client.get("/ready").status_code == 503
            finally:
                broken = app.state.redis
                app.state.redis = healthy
                client.portal.call(broken.aclose)
    finally:
        get_settings.cache_clear()
