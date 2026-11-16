"""Small, transaction-neutral repositories for application records."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import select, tuple_, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.business import (AgentAssignment, AuditLog, ChatRequest, DeliveryStatus,
    HandoffRequest, HandoffStatus, Message, OutboxEvent, RequestStatus, SenderType,
    ServiceStatus, Session, User)


class DuplicateKeyError(ValueError):
    pass


async def _add(db: AsyncSession, row):
    # SAVEPOINT keeps the caller's transaction usable after a unique violation.
    try:
        async with db.begin_nested():
            db.add(row)
            await db.flush()
    except IntegrityError as exc:
        if getattr(exc.orig, "sqlstate", None) == "23505":
            raise DuplicateKeyError("unique constraint violated") from exc
        raise
    return row


def _page(query, model, cursor: tuple[datetime, uuid.UUID] | None, limit: int):
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    if cursor:
        query = query.where(tuple_(model.created_at, model.id) > cursor)
    return query.order_by(model.created_at, model.id).limit(limit)


class UserRepository:
    def __init__(self, db: AsyncSession): self.db = db
    async def create(self, external_id: str): return await _add(self.db, User(external_id=external_id))
    async def get(self, id: uuid.UUID): return await self.db.get(User, id)
    async def get_by_external_id(self, external_id: str):
        return await self.db.scalar(select(User).where(User.external_id == external_id))
    async def list(self, cursor=None, limit=50):
        return list((await self.db.scalars(_page(select(User), User, cursor, limit))).all())


class SessionRepository:
    def __init__(self, db: AsyncSession): self.db = db
    async def create(self, user_id: uuid.UUID, thread_id: str):
        return await _add(self.db, Session(user_id=user_id, thread_id=thread_id))
    async def get(self, id: uuid.UUID): return await self.db.get(Session, id)
    async def get_by_thread_id(self, thread_id: str):
        return await self.db.scalar(select(Session).where(Session.thread_id == thread_id))
    async def list(self, user_id: uuid.UUID, cursor=None, limit=50):
        q = select(Session).where(Session.user_id == user_id)
        return list((await self.db.scalars(_page(q, Session, cursor, limit))).all())
    async def transition(self, id: uuid.UUID, expected: ServiceStatus, target: ServiceStatus) -> bool:
        result = await self.db.execute(update(Session).where(Session.id == id, Session.service_status == expected)
            .values(service_status=target, updated_at=datetime.now(timezone.utc)))
        return result.rowcount == 1


class MessageRepository:
    def __init__(self, db: AsyncSession): self.db = db
    async def create(self, session_id: uuid.UUID, sender_type: SenderType, content: str, sequence_no: int,
                     request_id: uuid.UUID | None = None):
        return await _add(self.db, Message(session_id=session_id, sender_type=sender_type,
                                           content=content, sequence_no=sequence_no,
                                           request_id=request_id))
    async def get(self, id: uuid.UUID): return await self.db.get(Message, id)
    async def list(self, session_id: uuid.UUID, after_sequence: int = 0, limit: int = 50):
        if not 1 <= limit <= 100: raise ValueError("limit must be between 1 and 100")
        q = select(Message).where(Message.session_id == session_id, Message.sequence_no > after_sequence)
        return list((await self.db.scalars(q.order_by(Message.sequence_no, Message.id).limit(limit))).all())


class ChatRequestRepository:
    def __init__(self, db: AsyncSession): self.db = db
    async def create(self, session_id: uuid.UUID, idempotency_key: str,
                     request_content: str | None = None):
        return await _add(self.db, ChatRequest(session_id=session_id, idempotency_key=idempotency_key,
                                               request_content=request_content))
    async def get(self, id: uuid.UUID): return await self.db.get(ChatRequest, id)
    async def get_by_key(self, session_id: uuid.UUID, key: str):
        return await self.db.scalar(select(ChatRequest).where(ChatRequest.session_id == session_id,
                                                             ChatRequest.idempotency_key == key))
    async def list(self, session_id: uuid.UUID, cursor=None, limit=50):
        return list((await self.db.scalars(_page(select(ChatRequest).where(ChatRequest.session_id == session_id),
                                                 ChatRequest, cursor, limit))).all())
    async def transition(self, id: uuid.UUID, expected: RequestStatus, target: RequestStatus,
                         response: str | None = None, error_code: str | None = None,
                         error_message: str | None = None) -> bool:
        result = await self.db.execute(update(ChatRequest).where(ChatRequest.id == id,
            ChatRequest.execution_status == expected).values(execution_status=target,
            response=response, error_code=error_code, error_message=error_message,
            updated_at=datetime.now(timezone.utc)))
        return result.rowcount == 1


class HandoffRequestRepository:
    def __init__(self, db: AsyncSession): self.db = db
    async def create(self, session_id: uuid.UUID, reason: str):
        return await _add(self.db, HandoffRequest(session_id=session_id, reason=reason))
    async def get(self, id: uuid.UUID): return await self.db.get(HandoffRequest, id)
    async def list(self, session_id: uuid.UUID, cursor=None, limit=50):
        return list((await self.db.scalars(_page(select(HandoffRequest).where(HandoffRequest.session_id == session_id),
                                                 HandoffRequest, cursor, limit))).all())
    async def transition(self, id: uuid.UUID, expected: HandoffStatus, target: HandoffStatus) -> bool:
        result = await self.db.execute(update(HandoffRequest).where(HandoffRequest.id == id,
            HandoffRequest.status == expected).values(status=target, updated_at=datetime.now(timezone.utc)))
        return result.rowcount == 1


class AgentAssignmentRepository:
    def __init__(self, db: AsyncSession): self.db = db
    async def create(self, session_id: uuid.UUID, agent_id: str):
        return await _add(self.db, AgentAssignment(session_id=session_id, agent_id=agent_id))
    async def get(self, id: uuid.UUID): return await self.db.get(AgentAssignment, id)
    async def get_active(self, session_id: uuid.UUID):
        return await self.db.scalar(select(AgentAssignment).where(AgentAssignment.session_id == session_id,
                                                                  AgentAssignment.released_at.is_(None)))
    async def list(self, session_id: uuid.UUID, after_id: uuid.UUID | None = None, limit=50):
        if not 1 <= limit <= 100: raise ValueError("limit must be between 1 and 100")
        q = select(AgentAssignment).where(AgentAssignment.session_id == session_id)
        if after_id:
            cursor = await self.get(after_id)
            if cursor is None or cursor.session_id != session_id: raise ValueError("invalid cursor")
            q = q.where(tuple_(AgentAssignment.assigned_at, AgentAssignment.id) >
                        (cursor.assigned_at, cursor.id))
        return list((await self.db.scalars(q.order_by(AgentAssignment.assigned_at, AgentAssignment.id).limit(limit))).all())
    async def release(self, id: uuid.UUID) -> bool:
        result = await self.db.execute(update(AgentAssignment).where(AgentAssignment.id == id,
            AgentAssignment.released_at.is_(None)).values(released_at=datetime.now(timezone.utc)))
        return result.rowcount == 1


class AuditLogRepository:
    def __init__(self, db: AsyncSession): self.db = db
    async def create(self, actor_type: str, action: str, payload: dict, session_id: uuid.UUID | None = None):
        return await _add(self.db, AuditLog(actor_type=actor_type, action=action, payload=payload, session_id=session_id))
    async def get(self, id: uuid.UUID): return await self.db.get(AuditLog, id)
    async def list(self, session_id: uuid.UUID, cursor=None, limit=50):
        return list((await self.db.scalars(_page(select(AuditLog).where(AuditLog.session_id == session_id),
                                                 AuditLog, cursor, limit))).all())


class OutboxEventRepository:
    def __init__(self, db: AsyncSession): self.db = db
    async def create(self, event_key: str, aggregate_type: str, aggregate_id: uuid.UUID,
                     event_type: str, payload: dict):
        return await _add(self.db, OutboxEvent(event_key=event_key, aggregate_type=aggregate_type,
            aggregate_id=aggregate_id, event_type=event_type, payload=payload))
    async def get(self, id: uuid.UUID): return await self.db.get(OutboxEvent, id)
    async def list(self, cursor=None, limit=50):
        return list((await self.db.scalars(_page(select(OutboxEvent), OutboxEvent, cursor, limit))).all())
    async def claim_due(self, limit: int = 50):
        if not 1 <= limit <= 100: raise ValueError("limit must be between 1 and 100")
        q = (select(OutboxEvent).where(OutboxEvent.status.in_([DeliveryStatus.pending, DeliveryStatus.failed]),
             OutboxEvent.available_at <= datetime.now(timezone.utc))
             .order_by(OutboxEvent.available_at, OutboxEvent.id).limit(limit).with_for_update(skip_locked=True))
        rows = list((await self.db.scalars(q)).all())
        for row in rows:
            row.status = DeliveryStatus.delivering
            row.attempts += 1
        await self.db.flush()
        return rows
    async def mark_delivered(self, id: uuid.UUID) -> bool:
        result = await self.db.execute(update(OutboxEvent).where(OutboxEvent.id == id,
            OutboxEvent.status == DeliveryStatus.delivering).values(status=DeliveryStatus.delivered,
            delivered_at=datetime.now(timezone.utc), last_error=None))
        return result.rowcount == 1
    async def mark_retry(self, id: uuid.UUID, available_at: datetime, error: str) -> bool:
        if available_at.tzinfo is None: raise ValueError("available_at must be timezone-aware UTC")
        result = await self.db.execute(update(OutboxEvent).where(OutboxEvent.id == id,
            OutboxEvent.status == DeliveryStatus.delivering).values(status=DeliveryStatus.failed,
            available_at=available_at.astimezone(timezone.utc), last_error=error))
        return result.rowcount == 1
