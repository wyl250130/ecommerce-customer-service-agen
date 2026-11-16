from backend.app.infrastructure.app_config import AppSettings
from backend.app.infrastructure.database import create_database_engine

import pytest


def test_settings_use_defaults():
    settings = AppSettings(_env_file=None)
    assert settings.app_name == "Customer Service"
    assert settings.environment == "local"


def test_settings_read_environment(monkeypatch):
    monkeypatch.setenv("CS_APP_NAME", "Configured Service")
    monkeypatch.setenv("CS_ENVIRONMENT", "test")
    settings = AppSettings(_env_file=None)
    assert settings.app_name == "Configured Service"
    assert settings.environment == "test"


def test_database_url_requires_async_postgresql():
    with pytest.raises(ValueError, match=r"postgresql\+asyncpg"):
        AppSettings(_env_file=None, database_url="sqlite+aiosqlite:///local.db")
    with pytest.raises(ValueError, match="CS_DATABASE_URL is required"):
        create_database_engine(AppSettings(_env_file=None, database_url=None))


def test_pool_configuration_is_validated():
    with pytest.raises(ValueError):
        AppSettings(_env_file=None, db_pool_size=0)
