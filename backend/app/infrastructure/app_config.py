"""Configuration for the application foundation."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CS_",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    app_name: str = "Customer Service"
    environment: str = "local"
    database_url: str | None = None
    test_database_url: str | None = None
    db_pool_size: int = Field(default=5, ge=1)
    db_max_overflow: int = Field(default=10, ge=0)
    db_pool_timeout: int = Field(default=30, ge=1)
    db_pool_recycle: int = Field(default=1800, ge=1)
    db_pool_pre_ping: bool = True
    redis_url: str | None = None
    redis_max_connections: int = Field(default=20, ge=1)
    redis_socket_timeout: float = Field(default=2.0, gt=0)
    redis_health_check_interval: int = Field(default=30, ge=0)
    redis_key_prefix: str = "cs"
    test_redis_url: str | None = None
    checkpoint_enabled: bool = False
    checkpoint_pool_min_size: int = Field(default=1, ge=0)
    checkpoint_pool_max_size: int = Field(default=10, ge=1)
    checkpoint_pool_timeout: float = Field(default=5.0, gt=0)
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llma_model: str | None = None
    llmb_model: str | None = None
    llm_timeout: float = Field(default=60.0, gt=0)
    dev_user_external_id: str | None = None
    default_snapshot_id: str | None = None
    facts_database_path: Path | None = None
    enable_manual_search: bool = False
    manual_source: Literal["frozen_v1", "legacy_sqlite"] = "frozen_v1"

    @field_validator("redis_url", "test_redis_url")
    @classmethod
    def validate_redis_url(cls, value: str | None) -> str | None:
        if not value:
            return None
        if not value.startswith(("redis://", "rediss://")):
            raise ValueError("Redis URL must use redis:// or rediss://")
        return value

    @field_validator("database_url", "test_database_url")
    @classmethod
    def validate_database_url(cls, value: str | None) -> str | None:
        if not value:
            return None
        try:
            url = make_url(value)
        except Exception as exc:
            raise ValueError("Database URL must be a valid SQLAlchemy URL") from exc
        if url.drivername != "postgresql+asyncpg" or not url.database:
            raise ValueError("Database URL must use postgresql+asyncpg and name a database")
        return value


@lru_cache
def get_settings() -> AppSettings:
    return AppSettings()
