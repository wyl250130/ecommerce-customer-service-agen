"""Atomic fixed-window rate limit based on Redis server time."""

from dataclasses import dataclass
from redis.asyncio import Redis
from redis.exceptions import RedisError
from .redis import RedisUnavailableError

_SCRIPT = """
local t=redis.call('TIME')
local now=t[1]*1000+math.floor(t[2]/1000)
local window=tonumber(ARGV[1])
local bucket=math.floor(now/window)
local key=KEYS[1]..':'..bucket
local count=redis.call('INCR',key)
if count==1 then redis.call('PEXPIRE',key,window+1000) end
return {count,(bucket+1)*window-now}
"""


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    remaining: int
    retry_after_ms: int
    limit: int


class RateLimiter:
    def __init__(self, redis: Redis, prefix: str = "cs"):
        self.redis, self.prefix = redis, prefix

    async def check(self, user_id: str, endpoint: str, *, limit: int, window_ms: int) -> RateLimitResult:
        if not user_id or not endpoint or any(":" in value for value in (user_id, endpoint)):
            raise ValueError("user and endpoint must be nonempty namespace components without ':'")
        if limit < 1 or window_ms < 1: raise ValueError("limit and window_ms must be positive")
        try:
            count, reset = await self.redis.eval(_SCRIPT, 1,
                f"{self.prefix}:rate:{endpoint}:{user_id}", window_ms)
        except RedisError as exc:
            raise RedisUnavailableError("Redis rate limiter failed") from exc
        return RateLimitResult(count <= limit, max(0, limit-count), int(reset) if count > limit else 0, limit)
