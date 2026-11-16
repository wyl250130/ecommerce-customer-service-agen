from uuid import UUID

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select

from backend.app.models.business import ChatRequest, Session, User
from backend.app.schemas.http_chat import (ChatRequestResponse, CreateSessionResponse, SessionResponse,
    MessageResponse, SendMessageBody)
from backend.app.services.application import ChatError, ChatService

router = APIRouter(prefix="/api")


def actor_identity(request: Request) -> str:
    settings = request.app.state.settings
    if settings.environment not in ("local", "test") or not settings.dev_user_external_id:
        raise ChatError(503, "identity_unavailable", "Authentication is not configured")
    return settings.dev_user_external_id


def service(request: Request) -> ChatService:
    if request.app.state.session_factory is None:
        raise ChatError(503, "database_unavailable", "Database is not configured")
    return ChatService(request.app)


@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session(actor: str = Depends(actor_identity), chat: ChatService = Depends(service)):
    row = await chat.sessions.create(actor)
    return CreateSessionResponse(id=row.id, service_status=row.service_status.value)


@router.get("/sessions", response_model=list[SessionResponse])
async def list_sessions(actor: str = Depends(actor_identity), chat: ChatService = Depends(service)):
    async with chat.factory() as db:
        rows = (await db.scalars(select(Session).join(User, User.id == Session.user_id)
                 .where(User.external_id == actor)
                 .order_by(Session.created_at.desc(), Session.id.desc()).limit(100))).all()
        return [SessionResponse(id=row.id, service_status=row.service_status.value,
                                created_at=row.created_at.isoformat()) for row in rows]


@router.post("/sessions/{session_id}/messages", response_model=ChatRequestResponse)
async def send_message(session_id: UUID, body: SendMessageBody,
                       actor: str = Depends(actor_identity), chat: ChatService = Depends(service)):
    row = await chat.send(session_id, actor, body.content, body.idempotency_key)
    return ChatRequestResponse.from_row(row)


@router.get("/sessions/{session_id}/messages", response_model=list[MessageResponse])
async def list_messages(session_id: UUID, after_sequence: int = 0, limit: int = 50,
                        actor: str = Depends(actor_identity), chat: ChatService = Depends(service)):
    if after_sequence < 0 or not 1 <= limit <= 100:
        raise ChatError(422, "invalid_pagination", "Invalid message cursor or limit")
    async with chat.factory() as db:
        await chat.sessions.authorized(db, session_id, actor)
        rows = await chat.messages.list(db, session_id, after_sequence, limit)
        return [MessageResponse.from_row(row) for row in rows]


@router.get("/chat-requests/{request_id}", response_model=ChatRequestResponse)
async def get_chat_request(request_id: UUID, actor: str = Depends(actor_identity),
                           chat: ChatService = Depends(service)):
    async with chat.factory() as db:
        row = await db.scalar(select(ChatRequest).join(Session, Session.id == ChatRequest.session_id)
                              .join(User, User.id == Session.user_id)
                              .where(ChatRequest.id == request_id, User.external_id == actor))
        if row is None:
            raise ChatError(404, "request_not_found", "Request not found")
        return ChatRequestResponse.from_row(row)


@router.get("/sessions/{session_id}/chat-requests/by-key/{key}",
            response_model=ChatRequestResponse)
async def get_request_by_key(session_id: UUID, key: str,
                             actor: str = Depends(actor_identity),
                             chat: ChatService = Depends(service)):
    async with chat.factory() as db:
        await chat.sessions.authorized(db, session_id, actor)
        row = await db.scalar(select(ChatRequest).where(
            ChatRequest.session_id == session_id, ChatRequest.idempotency_key == key))
        if row is None:
            raise ChatError(404, "request_not_found", "Request not found")
        return ChatRequestResponse.from_row(row)
