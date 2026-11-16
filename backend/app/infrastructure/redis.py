"""Shared async Redis client lifecycle."""

from redis.asyncio import Redis
from redis.exceptions import RedisError

from .app_config import AppSettings


class RedisUnavailableError(RuntimeError):
    pass


def create_redis_client(settings: AppSettings) -> Redis:
    if not settings.redis_url:
        raise ValueError("CS_REDIS_URL is required")
    return Redis.from_url(
        settings.redis_url, decode_responses=True,
        max_connections=settings.redis_max_connections,
        socket_timeout=settings.redis_socket_timeout,
        socket_connect_timeout=settings.redis_socket_timeout,
        health_check_interval=settings.redis_health_check_interval,
    )


async def check_redis(client: Redis) -> bool:
    try:
        return bool(await client.ping())
    except RedisError as exc:
        raise RedisUnavailableError("Redis is unavailable") from exc


async def close_redis(client: Redis) -> None:
    await client.aclose()
