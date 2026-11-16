"""Short business transactions around one guarded LangGraph execution."""

from uuid import UUID, uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import async_sessionmaker

from agent.context import RuntimeContext
from agent.handoff import SimulatedHandoffClient
from agent.state import make_initial_state
from agent.tools.manual import ManualSearchService
from backend.app.models.business import (ChatRequest, Message, RequestStatus,
    SenderType, ServiceStatus, Session, User)
from backend.app.repositories.sqlite import connection_factory
from backend.app.infrastructure.config import DB_PATH


class ChatError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        self.status_code, self.code, self.message = status_code, code, message


class SessionService:
    def __init__(self, factory: async_sessionmaker):
        self.factory = factory

    async def create(self, actor: str) -> Session:
        async with self.factory.begin() as db:
            await db.execute(insert(User).values(external_id=actor)
                             .on_conflict_do_nothing(index_elements=[User.external_id]))
            user = await db.scalar(select(User).where(User.external_id == actor))
            row = Session(user_id=user.id, thread_id=f"session-{uuid4()}")
            db.add(row)
            await db.flush()
            return row

    async def authorized(self, db, session_id: UUID, actor: str,
                         for_update: bool = False) -> Session:
        query = (select(Session).join(User, User.id == Session.user_id)
                 .where(Session.id == session_id, User.external_id == actor))
        if for_update:
            query = query.with_for_update(of=Session)
        row = await db.scalar(query)
        if row is None:
            raise ChatError(404, "session_not_found", "Session not found")
        return row


class MessageService:
    async def append(self, db, session_id: UUID, request_id: UUID,
                     sender: SenderType, content: str) -> Message:
        existing = await db.scalar(select(Message).where(
            Message.request_id == request_id, Message.sender_type == sender))
        if existing is not None:
            return existing
        sequence = await db.scalar(select(func.coalesce(func.max(Message.sequence_no), 0))
                                   .where(Message.session_id == session_id))
        row = Message(session_id=session_id, request_id=request_id,
                      sender_type=sender, content=content, sequence_no=sequence + 1)
        db.add(row)
        await db.flush()
        return row

    async def list(self, db, session_id: UUID, after_sequence: int, limit: int):
        return list((await db.scalars(select(Message).where(
            Message.session_id == session_id, Message.sequence_no > after_sequence)
            .order_by(Message.sequence_no, Message.id).limit(limit))).all())


class ChatService:
    def __init__(self, app):
        self.app = app
        self.factory = app.state.session_factory
        self.sessions = SessionService(self.factory)
        self.messages = MessageService()

    def context(self, session_id: UUID) -> RuntimeContext:
        settings = self.app.state.settings
        snapshot = settings.default_snapshot_id
        if not snapshot:
            raise ChatError(503, "snapshot_not_configured", "Server snapshot is not configured")
        return RuntimeContext(
            connection_factory=connection_factory(settings.facts_database_path or DB_PATH),
            default_snapshot_id=snapshot, visible_snapshot_ids=(snapshot,),
            manual_service=self.app.state.manual_service,
            handoff_client=self.app.state.handoff_client,
            conversation_id=str(session_id),
        )

    @staticmethod
    def _answer(values, request_id: UUID):
        messages = values.get("messages", [])
        marker = f"request:{request_id}"
        positions = [i for i, m in enumerate(messages)
                     if isinstance(m, HumanMessage) and m.id == marker]
        if not positions:
            return None
        tail = messages[positions[-1] + 1:]
        status = values.get("service_status", "active")
        if status == "transferred" and tail:
            return "已提交人工客服模拟工单。", "handoff", ServiceStatus.transferred
        if status == "handoff_failed" and tail:
            return "人工客服模拟接管失败，请稍后重试。", "fallback", ServiceStatus.active
        if tail and isinstance(tail[-1], ToolMessage):
            return "当前无法完成自动答复，请联系人工客服。", "fallback", ServiceStatus.active
        if tail and isinstance(tail[-1], AIMessage) and not tail[-1].tool_calls:
            # A direct LLM A reply asks for clarification; a final LLM B reply
            # follows finish_collection and its ToolMessage.
            kind = ("completed" if any(isinstance(m, ToolMessage)
                    and m.name == "finish_collection" for m in tail)
                    else "needs_user")
            content = tail[-1].content
            return str(content), kind, ServiceStatus.active
        return None

    async def _finalize(self, session_id: UUID, request_id: UUID, answer):
        content, kind, service_status = answer
        async with self.factory.begin() as db:
            session = await db.scalar(select(Session).where(Session.id == session_id)
                                      .with_for_update())
            request = await db.scalar(select(ChatRequest).where(ChatRequest.id == request_id)
                                      .with_for_update())
            if request.execution_status == RequestStatus.succeeded:
                return request
            if request.execution_status != RequestStatus.running:
                raise ChatError(409, "request_not_running", "Request cannot be finalized")
            await self.messages.append(db, session_id, request_id, SenderType.assistant, content)
            request.execution_status = RequestStatus.succeeded
            request.response = content
            request.result_type = kind
            session.service_status = service_status
            await db.flush()
            return request

    async def _recover(self, session: Session, request: ChatRequest):
        # The guard's completed status is written only after Graph.ainvoke
        # returns. A terminal checkpoint also covers a crash just before that
        # guard transition. Acquire the same two locks before reconciliation.
        from backend.app.infrastructure.execution_guard import ExecutionGuard, ExecutionBusyError
        from backend.app.infrastructure.locks import LockTimeoutError
        lock = self.app.state.session_lock
        if lock is None:
            raise ChatError(503, "chat_unavailable", "Chat execution unavailable")
        try:
            async with lock.hold(session.thread_id, wait_timeout=0.1) as lease:
                guard = await ExecutionGuard.acquire(self.app.state.settings.database_url,
                    session.thread_id, str(request.id), lease.owner_token, wait_timeout=0.1)
                try:
                    snapshot = await self.app.state.graph.aget_state(
                        {"configurable": {"thread_id": session.thread_id}})
                    answer = (self._answer(snapshot.values, request.id)
                              if snapshot and not snapshot.next else None)
                    if answer is None:
                        await guard.fail()
                        await self._mark_failed(request.id, "interrupted")
                        return await self.get_request(request.id)
                    await guard.complete()
                    return await self._finalize(session.id, request.id, answer)
                finally:
                    await guard.close()
        except (ExecutionBusyError, LockTimeoutError):
            return request

    async def _mark_failed(self, request_id: UUID, code: str):
        async with self.factory.begin() as db:
            row = await db.get(ChatRequest, request_id, with_for_update=True)
            if row and row.execution_status == RequestStatus.running:
                row.execution_status = RequestStatus.failed
                row.error_code = code
                row.error_message = "Chat execution did not complete"

    async def get_request(self, request_id: UUID):
        async with self.factory() as db:
            return await db.get(ChatRequest, request_id)

    async def send(self, session_id: UUID, actor: str, content: str, key: str):
        if self.app.state.invoke_graph is None:
            raise ChatError(503, "chat_unavailable", "Chat execution unavailable")
        context = self.context(session_id)
        async with self.factory.begin() as db:
            session = await self.sessions.authorized(db, session_id, actor, for_update=True)
            request = await db.scalar(select(ChatRequest).where(
                ChatRequest.session_id == session_id,
                ChatRequest.idempotency_key == key))
            if request is not None:
                if request.request_content != content:
                    raise ChatError(409, "idempotency_conflict", "Key belongs to another message")
                duplicate = True
            else:
                if session.service_status != ServiceStatus.active:
                    raise ChatError(409, "session_unavailable", "Session is not served by AI")
                pending = await db.scalar(select(ChatRequest.id).where(
                    ChatRequest.session_id == session_id,
                    ChatRequest.execution_status.in_((RequestStatus.pending, RequestStatus.running))))
                if pending is not None:
                    raise ChatError(409, "session_busy", "Another request needs completion")
                request = ChatRequest(session_id=session_id, idempotency_key=key,
                                      request_content=content, execution_status=RequestStatus.running)
                db.add(request)
                await db.flush()
                await self.messages.append(db, session_id, request.id, SenderType.user, content)
                duplicate = False
            thread_id, request_id = session.thread_id, request.id
        if duplicate:
            if request.execution_status == RequestStatus.running:
                return await self._recover(session, request)
            return request

        config = {"configurable": {"thread_id": thread_id}}
        try:
            saved = await self.app.state.graph.aget_state(config)
            human = HumanMessage(content=content, id=f"request:{request_id}")
            state = ({"messages": [human]} if saved and saved.values else
                     make_initial_state([human], context=context))
            result = await self.app.state.invoke_graph(state, config=config,
                context=context, request_id=str(request_id))
            answer = self._answer(result, request_id)
            if answer is None:
                raise ValueError("Graph returned no user-visible result")
        except Exception as exc:
            await self._mark_failed(request_id, "graph_failed")
            raise ChatError(503, "graph_failed", "Chat execution failed") from exc
        return await self._finalize(session_id, request_id, answer)
