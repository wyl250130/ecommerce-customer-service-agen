from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import text
from redis.exceptions import RedisError

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request) -> dict[str, str]:
    engine = getattr(request.app.state, "db_engine", None)
    redis_client = getattr(request.app.state, "redis", None)
    if engine is None or redis_client is None:
        raise HTTPException(status_code=503, detail="Dependencies are not configured")
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        await redis_client.ping()
    except (Exception, RedisError) as exc:
        raise HTTPException(status_code=503, detail="Dependency unavailable") from exc
    return {"status": "ready"}
