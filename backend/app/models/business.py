"""Application persistence models. Agent tool messages and evidence stay outside these tables."""

import enum
import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, Enum, ForeignKey, Index, Integer, String, Text, UniqueConstraint, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class ServiceStatus(str, enum.Enum):
    active = "active"
    handoff_pending = "handoff_pending"
    transferred = "transferred"
    closed = "closed"


class SenderType(str, enum.Enum):
    user = "user"
    assistant = "assistant"
    human_agent = "human_agent"
    system = "system"


class RequestStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class HandoffStatus(str, enum.Enum):
    pending = "pending"
    assigned = "assigned"
    completed = "completed"
    cancelled = "cancelled"


class DeliveryStatus(str, enum.Enum):
    pending = "pending"
    delivering = "delivering"
    delivered = "delivered"
    failed = "failed"


def enum_type(cls: type[enum.Enum]) -> Enum:
    return Enum(cls, native_enum=False, create_constraint=True, validate_strings=True,
                name=f"ck_{cls.__name__.lower()}")


class Timestamped:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class User(Base, Timestamped):
    __tablename__ = "users"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    external_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)


class Session(Base, Timestamped):
    __tablename__ = "sessions"
    __table_args__ = (Index("ix_sessions_user_created_id", "user_id", "created_at", "id"),)
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    thread_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    service_status: Mapped[ServiceStatus] = mapped_column(enum_type(ServiceStatus), default=ServiceStatus.active, nullable=False)


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (UniqueConstraint("session_id", "sequence_no", name="uq_messages_session_sequence"),
                      UniqueConstraint("request_id", "sender_type", name="uq_messages_request_sender"))
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False, index=True)
    request_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("chat_requests.id"), index=True)
    sender_type: Mapped[SenderType] = mapped_column(enum_type(SenderType), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sequence_no: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class ChatRequest(Base, Timestamped):
    __tablename__ = "chat_requests"
    __table_args__ = (UniqueConstraint("session_id", "idempotency_key", name="uq_chat_requests_idempotency"),
                      Index("ix_chat_requests_session_created_id", "session_id", "created_at", "id"))
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    request_content: Mapped[str | None] = mapped_column(Text)
    result_type: Mapped[str | None] = mapped_column(String(32))
    execution_status: Mapped[RequestStatus] = mapped_column(enum_type(RequestStatus), default=RequestStatus.pending, nullable=False)
    response: Mapped[str | None] = mapped_column(Text)
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(Text)


class HandoffRequest(Base, Timestamped):
    __tablename__ = "handoff_requests"
    __table_args__ = (Index("ix_handoff_requests_session_created_id", "session_id", "created_at", "id"),)
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[HandoffStatus] = mapped_column(enum_type(HandoffStatus), default=HandoffStatus.pending, nullable=False)


class AgentAssignment(Base):
    __tablename__ = "agent_assignments"
    __table_args__ = (Index("uq_agent_assignments_active_session", "session_id", unique=True,
                            postgresql_where=text("released_at IS NULL")),
                      Index("ix_agent_assignments_session_assigned_id", "session_id", "assigned_at", "id"),
                      CheckConstraint("released_at IS NULL OR released_at >= assigned_at", name="ck_agent_assignments_release_time"))
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False)
    agent_id: Mapped[str] = mapped_column(String(128), nullable=False)
    assigned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    released_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AuditLog(Base):
    __tablename__ = "audit_logs"
    __table_args__ = (Index("ix_audit_logs_session_created_id", "session_id", "created_at", "id"),)
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("sessions.id"))
    actor_type: Mapped[str] = mapped_column(String(32), nullable=False)
    action: Mapped[str] = mapped_column(String(128), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class OutboxEvent(Base):
    __tablename__ = "outbox_events"
    __table_args__ = (Index("ix_outbox_events_due", "status", "available_at", "id"),
                      Index("ix_outbox_events_created_id", "created_at", "id"),
                      CheckConstraint("attempts >= 0", name="ck_outbox_events_attempts"))
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_key: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    aggregate_type: Mapped[str] = mapped_column(String(64), nullable=False)
    aggregate_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    event_type: Mapped[str] = mapped_column(String(128), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    status: Mapped[DeliveryStatus] = mapped_column(enum_type(DeliveryStatus), default=DeliveryStatus.pending, nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
