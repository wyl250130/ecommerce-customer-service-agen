"""Real HTTP, PostgreSQL, Redis, checkpoint and existing business-tool path."""

import asyncio
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from sqlalchemy import delete, select

from backend.app.infrastructure.checkpoint import open_checkpointer
from backend.app.infrastructure.database import create_database_engine, create_session_factory
from backend.app.models.business import ChatRequest, Message, Session, User
from backend.app.models.execution_guard import ExecutionGuard as GuardRow

from backend.app.api.chat import actor_identity
from backend.app.infrastructure.app_config import AppSettings, get_settings
from backend.app.main import create_app
from backend.app.services.application import ChatService


ROOT = Path(__file__).resolve().parents[2]
FACTS = ROOT / "data/processed/customer_service/customer_service_train.sqlite"


class ScriptedA:
    calls = 0
    delay = 0
    started = threading.Event()
    human_counts = []

    def __init__(self, order_id):
        self.order_id = order_id

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        type(self).calls += 1
        type(self).human_counts.append(sum(isinstance(m, HumanMessage) for m in messages))
        if self.delay:
            self.started.set()
            time.sleep(self.delay)
        last = messages[-1]
        if isinstance(last, ToolMessage):
            return AIMessage(content="", tool_calls=[{
                "name": "finish_collection", "args": {"evidence_indices": [0]},
                "id": f"finish-{uuid4()}"}])
        latest_user = next(m.content for m in reversed(messages)
                           if isinstance(m, HumanMessage))
        if "人工" in latest_user:
            name, args = "handoff_to_human", {
                "reason_code": "user_requested", "reason_detail": "用户要求转人工"}
        elif "兜底" in latest_user:
            name, args = "request_fallback", {"reason_code": "unavailable"}
        elif "追问" in latest_user:
            return AIMessage(content="请补充需要查询的订单信息。")
        elif "失败" in latest_user:
            raise TimeoutError("scripted model timeout")
        else:
            name, args = "get_order_info", {"order_id": self.order_id}
        return AIMessage(content="", tool_calls=[{
            "name": name, "args": args, "id": f"tool-{uuid4()}"}])


class ScriptedB:
    def invoke(self, messages):
        assert "order_id" in messages[-1].content
        return AIMessage(content="已查询订单状态。")


@pytest.fixture
def chat_app(monkeypatch):
    settings = AppSettings()
    if not settings.test_database_url or not settings.test_redis_url:
        pytest.skip("test PostgreSQL and Redis URLs are required")
    with sqlite3.connect(FACTS) as db:
        snapshot, order_id = db.execute(
            "SELECT snapshot_id, order_id FROM orders LIMIT 1").fetchone()
    user = f"http-test-{uuid4()}"
    monkeypatch.setenv("CS_DATABASE_URL", settings.test_database_url)
    monkeypatch.setenv("CS_REDIS_URL", settings.test_redis_url)
    monkeypatch.setenv("CS_CHECKPOINT_ENABLED", "true")
    monkeypatch.setenv("CS_ENVIRONMENT", "test")
    monkeypatch.setenv("CS_DEV_USER_EXTERNAL_ID", user)
    monkeypatch.setenv("CS_DEFAULT_SNAPSHOT_ID", snapshot)
    monkeypatch.setenv("CS_FACTS_DATABASE_PATH", str(FACTS))
    monkeypatch.setenv("CS_REDIS_KEY_PREFIX", f"test:{uuid4()}")
    get_settings.cache_clear()
    ScriptedA.calls, ScriptedA.delay = 0, 0
    ScriptedA.started.clear()
    ScriptedA.human_counts.clear()
    app = create_app(model_factory=lambda: (ScriptedA(order_id), ScriptedB()))
    yield app, order_id

    async def cleanup():
        config = AppSettings(_env_file=None, database_url=settings.test_database_url)
        engine = create_database_engine(config)
        factory = create_session_factory(engine)
        try:
            async with factory.begin() as db:
                owner = await db.scalar(select(User).where(User.external_id == user))
                if owner is None:
                    return
                sessions = list((await db.scalars(select(Session).where(
                    Session.user_id == owner.id))).all())
                ids = [row.id for row in sessions]
                threads = [row.thread_id for row in sessions]
                resources = await open_checkpointer(config)
                try:
                    for thread in threads:
                        await resources.saver.adelete_thread(thread)
                finally:
                    await resources.close()
                if ids:
                    await db.execute(delete(Message).where(Message.session_id.in_(ids)))
                    await db.execute(delete(ChatRequest).where(ChatRequest.session_id.in_(ids)))
                    await db.execute(delete(Session).where(Session.id.in_(ids)))
                    await db.execute(delete(GuardRow).where(GuardRow.thread_id.in_(threads)))
                await db.delete(owner)
        finally:
            await engine.dispose()

    asyncio.run(cleanup())
    get_settings.cache_clear()


def test_http_chat_tool_multiturn_restart_and_recovery(chat_app, monkeypatch):
    app, order_id = chat_app
    with TestClient(app, raise_server_exceptions=False) as client:
        created = client.post("/api/sessions")
        assert created.status_code == 200, created.text
        session_id = created.json()["id"]
        assert session_id in [item["id"] for item in client.get("/api/sessions").json()]
        body = {"content": f"查询订单 {order_id}", "idempotency_key": "first"}
        first = client.post(f"/api/sessions/{session_id}/messages", json=body)
        assert first.status_code == 200, first.text
        assert first.json()["result_type"] == "completed"
        assert first.json()["response"] == "已查询订单状态。"
        count = ScriptedA.calls
        duplicate = client.post(f"/api/sessions/{session_id}/messages", json=body)
        assert duplicate.json()["id"] == first.json()["id"]
        assert ScriptedA.calls == count
        assert client.post(f"/api/sessions/{session_id}/messages", json={
            "content": "other", "idempotency_key": "first"}).status_code == 409
        assert client.get(f"/api/chat-requests/{first.json()['id']}").json()[
            "execution_status"] == "succeeded"
        assert client.get(f"/api/sessions/{session_id}/chat-requests/by-key/first").json()[
            "id"] == first.json()["id"]
        assert client.get(f"/api/sessions/{session_id}/chat-requests/by-key/missing").status_code == 404
        messages = client.get(f"/api/sessions/{session_id}/messages").json()
        assert [(m["sender_type"], m["sequence_no"]) for m in messages] == [
            ("user", 1), ("assistant", 2)]
        app.dependency_overrides[actor_identity] = lambda: "another-user"
        assert session_id not in [item["id"] for item in client.get("/api/sessions").json()]
        assert client.get(f"/api/sessions/{session_id}/messages").status_code == 404
        assert client.get(f"/api/chat-requests/{first.json()['id']}").status_code == 404
        assert client.get(f"/api/sessions/{session_id}/chat-requests/by-key/first").status_code == 404
        app.dependency_overrides.clear()

        original = ChatService._finalize
        fail_once = {"value": True}

        async def failing_finalize(self, *args):
            if fail_once["value"]:
                fail_once["value"] = False
                raise RuntimeError("simulated business write failure")
            return await original(self, *args)

        monkeypatch.setattr(ChatService, "_finalize", failing_finalize)
        failed_persist = client.post(f"/api/sessions/{session_id}/messages", json={
            "content": "追问", "idempotency_key": "recover"})
        assert failed_persist.status_code == 500
        calls_after_graph = ScriptedA.calls
        recovered = client.post(f"/api/sessions/{session_id}/messages", json={
            "content": "追问", "idempotency_key": "recover"})
        assert recovered.status_code == 200, recovered.text
        assert recovered.json()["result_type"] == "needs_user"
        assert ScriptedA.calls == calls_after_graph
        monkeypatch.setattr(ChatService, "_finalize", original)

        fallback = client.post(f"/api/sessions/{session_id}/messages", json={
            "content": "请兜底", "idempotency_key": "fallback"})
        assert fallback.status_code == 200, fallback.text
        assert fallback.json()["result_type"] == "fallback"

    app2 = create_app(model_factory=lambda: (ScriptedA(order_id), ScriptedB()))
    with TestClient(app2) as client:
        restarted = client.post(f"/api/sessions/{session_id}/messages", json={
            "content": "再次查询订单", "idempotency_key": "restart"})
        assert restarted.status_code == 200, restarted.text
        assert restarted.json()["result_type"] == "completed"
        assert max(ScriptedA.human_counts) >= 4
        handoff = client.post(f"/api/sessions/{session_id}/messages", json={
            "content": "我要人工", "idempotency_key": "handoff"})
        assert handoff.status_code == 200, handoff.text
        assert handoff.json()["result_type"] == "handoff"
        assert client.post(f"/api/sessions/{session_id}/messages", json={
            "content": "继续", "idempotency_key": "after-handoff"}).status_code == 409


def test_http_chat_failure_and_same_session_busy(chat_app):
    app, _ = chat_app
    with TestClient(app) as client:
        session_id = client.post("/api/sessions").json()["id"]
        failed = client.post(f"/api/sessions/{session_id}/messages", json={
            "content": "请模拟失败", "idempotency_key": "failed"})
        assert failed.status_code == 503
        # Graph failures remain queryable; duplicate keys never rerun the model.
        rows = client.get(f"/api/sessions/{session_id}/messages").json()
        assert len(rows) == 1 and rows[0]["sender_type"] == "user"
        request_id = rows[0]["request_id"]
        assert client.get(f"/api/chat-requests/{request_id}").json()["execution_status"] == "failed"

        ScriptedA.delay = 0.3
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(client.post, f"/api/sessions/{session_id}/messages",
                json={"content": "查询订单", "idempotency_key": "slow"})
            assert ScriptedA.started.wait(timeout=2)
            contender = client.post(f"/api/sessions/{session_id}/messages", json={
                "content": "另一条请求", "idempotency_key": "contender"})
            assert contender.status_code == 409
            assert contender.json()["error"] == "session_busy"
            assert first.result(timeout=5).status_code == 200
        ScriptedA.delay = 0
