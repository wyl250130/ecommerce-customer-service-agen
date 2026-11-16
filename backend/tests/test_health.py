from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.infrastructure.app_config import get_settings


def test_health_endpoint(monkeypatch):
    monkeypatch.setenv("CS_APP_NAME", "Test Customer Service")
    monkeypatch.setenv("CS_CHECKPOINT_ENABLED", "false")
    get_settings.cache_clear()
    try:
        app = create_app()
        with TestClient(app) as client:
            response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        assert app.title == "Test Customer Service"
    finally:
        get_settings.cache_clear()


def test_health_does_not_require_database_connection(monkeypatch):
    monkeypatch.setenv("CS_DATABASE_URL", "postgresql+asyncpg://localhost/unavailable")
    monkeypatch.setenv("CS_CHECKPOINT_ENABLED", "false")
    get_settings.cache_clear()
    try:
        app = create_app()
        with TestClient(app) as client:
            assert app.state.session_factory is not None
            response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
    finally:
        get_settings.cache_clear()
