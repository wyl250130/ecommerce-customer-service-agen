"""Run against a disposable PostgreSQL database set by CS_TEST_DATABASE_URL."""

import asyncio
from pathlib import Path
from uuid import uuid4

from alembic import command
from alembic.config import Config
from fastapi import Depends, HTTPException
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.infrastructure.app_config import AppSettings, get_settings
from backend.app.infrastructure.database import create_database_engine, get_session
from backend.app.main import create_app


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def test_url() -> str:
    url = AppSettings().test_database_url
    if not url:
        pytest.skip("Set CS_TEST_DATABASE_URL to a disposable PostgreSQL database")
    AppSettings(_env_file=None, database_url=url)
    return url


def test_postgresql_connection(test_url):
    async def check() -> None:
        engine = create_database_engine(AppSettings(_env_file=None, database_url=test_url))
        try:
            async with engine.connect() as connection:
                assert (await connection.scalar(text("SELECT 1"))) == 1
        finally:
            await engine.dispose()

    asyncio.run(check())


def test_request_transaction_commit_rollback_and_session_close(test_url, monkeypatch):
    monkeypatch.setenv("CS_DATABASE_URL", test_url)
    get_settings.cache_clear()
    table = f"cs_infra_test_{uuid4().hex}"
    app = create_app()

    @app.post("/test/setup")
    async def setup(session: AsyncSession = Depends(get_session)):
        await session.execute(text(f"CREATE TABLE {table} (id integer PRIMARY KEY)"))
        return {"ok": True}

    @app.post("/test/commit")
    async def commit(session: AsyncSession = Depends(get_session)):
        await session.execute(text(f"INSERT INTO {table} (id) VALUES (1)"))
        return {"ok": True}

    @app.post("/test/rollback")
    async def rollback(session: AsyncSession = Depends(get_session)):
        await session.execute(text(f"INSERT INTO {table} (id) VALUES (2)"))
        raise HTTPException(status_code=400, detail="force rollback")

    @app.get("/test/count")
    async def count(session: AsyncSession = Depends(get_session)):
        return {"count": await session.scalar(text(f"SELECT count(*) FROM {table}"))}

    @app.delete("/test/cleanup")
    async def cleanup(session: AsyncSession = Depends(get_session)):
        await session.execute(text(f"DROP TABLE IF EXISTS {table}"))
        return {"ok": True}

    try:
        with TestClient(app) as client:
            try:
                assert client.post("/test/setup").status_code == 200
                assert client.post("/test/commit").status_code == 200
                assert client.get("/test/count").json() == {"count": 1}
                assert client.post("/test/rollback").status_code == 400
                assert client.get("/test/count").json() == {"count": 1}
                assert app.state.db_engine.pool.checkedout() == 0
            finally:
                client.delete("/test/cleanup")
        assert app.state.db_engine.pool.checkedout() == 0
    finally:
        get_settings.cache_clear()


def test_alembic_upgrade_head(test_url, monkeypatch):
    monkeypatch.setenv("CS_DATABASE_URL", test_url)
    get_settings.cache_clear()
    try:
        command.upgrade(Config(str(ROOT / "alembic.ini")), "head")

        async def check_version() -> None:
            engine = create_database_engine(AppSettings(_env_file=None, database_url=test_url))
            try:
                async with engine.connect() as connection:
                    version = await connection.scalar(text("SELECT version_num FROM alembic_version"))
                    assert version == "95c87f0b4a22"
            finally:
                await engine.dispose()

        asyncio.run(check_version())
    finally:
        get_settings.cache_clear()
