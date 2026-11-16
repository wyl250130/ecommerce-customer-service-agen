"""Alembic environment for async PostgreSQL migrations."""

import asyncio

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import create_async_engine

from backend.app.infrastructure.app_config import get_settings
from backend.app.models import Base


config = context.config
target_metadata = Base.metadata


def include_name(name, type_, parent_names):
    # LangGraph owns checkpoint_* tables; Alembic must never propose dropping them.
    return not (type_ == "table" and name and
                (name == "checkpoints" or name.startswith("checkpoint_")))


def database_url() -> str:
    url = get_settings().database_url
    if url is None:
        raise RuntimeError("CS_DATABASE_URL is required for Alembic migrations")
    return url


def run_migrations_offline() -> None:
    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        include_name=include_name,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata,
                      compare_type=True, include_name=include_name)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_async_engine(database_url(), poolclass=pool.NullPool)
    try:
        async with engine.connect() as connection:
            await connection.run_sync(do_run_migrations)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
