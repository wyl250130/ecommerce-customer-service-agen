"""PostgreSQL contract tests for application repositories."""

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import delete
from sqlalchemy.exc import IntegrityError

from backend.app.infrastructure.app_config import AppSettings, get_settings
from backend.app.infrastructure.database import create_database_engine, create_session_factory
from backend.app.models.business import (AuditLog, ChatRequest, DeliveryStatus, HandoffRequest,
    HandoffStatus, Message, OutboxEvent, RequestStatus, SenderType, ServiceStatus,
    Session, User, AgentAssignment)
from backend.app.repositories.business import (AgentAssignmentRepository, AuditLogRepository,
    ChatRequestRepository, DuplicateKeyError, HandoffRequestRepository, MessageRepository,
    OutboxEventRepository, SessionRepository, UserRepository)


def test_business_repositories(monkeypatch):
    url = AppSettings().test_database_url
    if not url:
        pytest.skip("CS_TEST_DATABASE_URL is required")
    monkeypatch.setenv("CS_DATABASE_URL", url)
    get_settings.cache_clear()
    command.upgrade(Config(str(Path(__file__).resolve().parents[2] / "alembic.ini")), "head")
    get_settings.cache_clear()

    async def run():
        engine = create_database_engine(AppSettings(_env_file=None, database_url=url))
        factory = create_session_factory(engine)
        token = uuid.uuid4().hex
        try:
            async with factory.begin() as db:
                user = await UserRepository(db).create(f"test-{token}")
                conversation = await SessionRepository(db).create(user.id, f"thread-{token}")
                user_id, session_id = user.id, conversation.id
                assert conversation.created_at.tzinfo is not None

            async with factory.begin() as db:
                sessions = SessionRepository(db)
                assert (await sessions.get(session_id)).thread_id == f"thread-{token}"
                assert len(await sessions.list(user_id)) == 1
                messages = MessageRepository(db)
                await messages.create(session_id, SenderType.user, "first", 1)
                await messages.create(session_id, SenderType.assistant, "second", 2)
                with pytest.raises(DuplicateKeyError):
                    await messages.create(session_id, SenderType.user, "duplicate", 1)
                assert [m.sequence_no for m in await messages.list(session_id)] == [1, 2]
                assert [m.sequence_no for m in await messages.list(session_id, after_sequence=1)] == [2]
                requests = ChatRequestRepository(db)
                request = await requests.create(session_id, f"key-{token}")
                with pytest.raises(DuplicateKeyError):
                    await requests.create(session_id, f"key-{token}")
                assert (await requests.get_by_key(session_id, f"key-{token}")).id == request.id
                assignment_repo = AgentAssignmentRepository(db)
                assignment = await assignment_repo.create(session_id, "human-1")
                with pytest.raises(DuplicateKeyError):
                    await assignment_repo.create(session_id, "human-2")
                assert (await assignment_repo.get_active(session_id)).id == assignment.id
                assert await assignment_repo.release(assignment.id)
                assert not await assignment_repo.release(assignment.id)
                await assignment_repo.create(session_id, "human-2")
                handoff = await HandoffRequestRepository(db).create(session_id, "user requested")
                assert await HandoffRequestRepository(db).transition(handoff.id, HandoffStatus.pending, HandoffStatus.assigned)
                await AuditLogRepository(db).create("system", "handoff", {"ok": True}, session_id)
                event = await OutboxEventRepository(db).create(f"event-{token}", "session", session_id, "handoff", {})
                event_id, request_id = event.id, request.id
                assert (await AuditLogRepository(db).list(session_id))[0].action == "handoff"

            # A failed unit of work must not leave a row behind.
            with pytest.raises(RuntimeError):
                async with factory.begin() as db:
                    await UserRepository(db).create(f"rollback-{token}")
                    raise RuntimeError("rollback")
            async with factory.begin() as db:
                assert await UserRepository(db).get_by_external_id(f"rollback-{token}") is None
                with pytest.raises(IntegrityError):
                    await SessionRepository(db).create(uuid.uuid4(), f"bad-fk-{token}")

            async def try_transition():
                async with factory.begin() as db:
                    return await SessionRepository(db).transition(session_id, ServiceStatus.active, ServiceStatus.handoff_pending)
            assert sorted(await asyncio.gather(try_transition(), try_transition())) == [False, True]

            async with factory.begin() as db:
                requests = ChatRequestRepository(db)
                assert await requests.transition(request_id, RequestStatus.pending, RequestStatus.running)
                assert not await requests.transition(request_id, RequestStatus.pending, RequestStatus.failed)
                outbox = OutboxEventRepository(db)
                claimed = await outbox.claim_due()
                assert event_id in {row.id for row in claimed}
                assert await outbox.mark_retry(event_id, datetime.now(timezone.utc), "temporary")
            async with factory.begin() as db:
                outbox = OutboxEventRepository(db)
                claimed = await outbox.claim_due()
                assert event_id in {row.id for row in claimed}
                assert await outbox.mark_delivered(event_id)
                assert not await outbox.mark_delivered(event_id)
                assert (await outbox.get(event_id)).attempts == 2
                assert (await outbox.get(event_id)).status == DeliveryStatus.delivered
        finally:
            # Delete only rows created by this test, preserving other test data.
            if "session_id" in locals():
                async with factory.begin() as db:
                    for model in (Message, ChatRequest, HandoffRequest, AgentAssignment, AuditLog):
                        await db.execute(delete(model).where(model.session_id == session_id))
                    await db.execute(delete(OutboxEvent).where(OutboxEvent.aggregate_id == session_id))
                    await db.execute(delete(Session).where(Session.id == session_id))
                    await db.execute(delete(User).where(User.id == user_id))
            await engine.dispose()

    asyncio.run(run())
