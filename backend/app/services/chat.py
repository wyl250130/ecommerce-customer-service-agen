"""One turn of the existing LangGraph kernel."""

from contextlib import closing

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from agent.context import RuntimeContext
from agent.graph import build_agent
from agent.handoff import SimulatedHandoffClient
from agent.state import make_initial_state
from agent.tools.manual import ManualSearchService
from backend.app.infrastructure.config import DB_PATH, Settings
from backend.app.repositories.sqlite import connection_factory
from backend.app.schemas.chat import ChatRequest, ChatResponse


class ChatService:
    def __init__(self, settings: Settings):
        settings.validate_chat()
        self.settings = settings
        self.connect = connection_factory(DB_PATH)
        with closing(self.connect()) as conn:
            exists = conn.execute("SELECT 1 FROM orders WHERE snapshot_id=? LIMIT 1", (settings.default_snapshot_id,)).fetchone()
        if not exists:
            raise ValueError("DEFAULT_SNAPSHOT_ID is absent from the bundled database")
        model_kwargs = {"api_key": settings.llm_api_key}
        if settings.llm_base_url:
            model_kwargs["base_url"] = settings.llm_base_url
        llma = ChatOpenAI(model=settings.llma_model, **model_kwargs)
        llmb = ChatOpenAI(model=settings.llmb_model, **model_kwargs)
        self.graph = build_agent(llma, llmb)
        self.manual_service = ManualSearchService() if settings.enable_manual_search else None
        self.handoff_client = SimulatedHandoffClient()

    def reply(self, request: ChatRequest) -> ChatResponse:
        context = RuntimeContext(
            connection_factory=self.connect,
            default_snapshot_id=self.settings.default_snapshot_id,
            visible_snapshot_ids=(self.settings.default_snapshot_id,),
            manual_service=self.manual_service,
            handoff_client=self.handoff_client,
            conversation_id=request.conversation_id,
        )
        initial = make_initial_state([HumanMessage(content=request.message)], context=context)
        result = self.graph.invoke(initial, context=context, config={"recursion_limit": 32})
        messages = result["messages"]
        if messages and isinstance(messages[-1], ToolMessage):
            answer = ("已提交人工客服模拟工单。" if result.get("service_status") == "transferred"
                      else "当前无法完成自动答复，请联系人工客服。")
        else:
            answer = str(messages[-1].content) if messages else ""
        return ChatResponse(
            conversation_id=request.conversation_id,
            answer=answer,
            service_status=result.get("service_status", "active"),
            tool_call_count=result.get("tool_call_count", 0),
        )
