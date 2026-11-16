"""JSON cache for explicitly non-sensitive, reproducible values only."""

import json
from typing import Any
from redis.asyncio import Redis
from redis.exceptions import RedisError
from .redis import RedisUnavailableError


class JsonCache:
    def __init__(self, redis: Redis, prefix: str = "cs"):
        self.redis, self.prefix = redis, prefix

    def key(self, key: str) -> str:
        if not key: raise ValueError("cache key is required")
        return f"{self.prefix}:cache:{key}"

    async def get(self, key: str) -> Any | None:
        try:
            value = await self.redis.get(self.key(key))
        except RedisError as exc:
            raise RedisUnavailableError("Redis cache read failed") from exc
        return None if value is None else json.loads(value)

    async def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        if ttl_seconds < 1: raise ValueError("ttl_seconds must be positive")
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False)
        try:
            await self.redis.set(self.key(key), payload, ex=ttl_seconds)
        except RedisError as exc:
            raise RedisUnavailableError("Redis cache write failed") from exc

    async def delete(self, key: str) -> bool:
        try:
            return bool(await self.redis.delete(self.key(key)))
        except RedisError as exc:
            raise RedisUnavailableError("Redis cache delete failed") from exc
