from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes import router
from backend.app.api.chat import router as chat_router
from backend.app.infrastructure.app_config import get_settings
from backend.app.infrastructure.database import create_database_engine, create_session_factory
from backend.app.infrastructure.redis import close_redis, create_redis_client
from backend.app.infrastructure.checkpoint import GraphStateReader, open_checkpointer
from backend.app.infrastructure.execution_runner import run_owned_graph
from backend.app.infrastructure.locks import SessionLock
from backend.app.infrastructure.model_clients import create_model_clients
from backend.app.services.application import ChatError
from agent.handoff import SimulatedHandoffClient
from agent.tools.manual import ManualSearchService
from agent.tools.manuals.legacy import LegacyManualSearchService
from backend.app.infrastructure.config import DB_PATH


def create_app(model_factory=None) -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = create_database_engine(settings) if settings.database_url else None
        app.state.db_engine = engine
        app.state.session_factory = create_session_factory(engine) if engine else None
        redis_client = create_redis_client(settings) if settings.redis_url else None
        app.state.redis = redis_client
        app.state.session_lock = None
        app.state.handoff_client = SimulatedHandoffClient()
        app.state.manual_service = None
        if settings.enable_manual_search:
            app.state.manual_service = (
                LegacyManualSearchService(settings.facts_database_path or DB_PATH)
                if settings.manual_source == "legacy_sqlite" else ManualSearchService())
        checkpoint = None
        app.state.checkpointer = None
        app.state.graph = None
        app.state.invoke_graph = None
        try:
            if settings.checkpoint_enabled:
                checkpoint = await open_checkpointer(settings)
                app.state.checkpointer = checkpoint.saver
                active_factory = model_factory or (create_model_clients if settings.llm_api_key else None)
                if active_factory is not None:
                    if redis_client is None:
                        raise RuntimeError("Redis is required for coordinated Graph execution")
                    from agent.graph import build_agent
                    llma, llmb = (active_factory(settings) if active_factory is create_model_clients
                                  else active_factory())
                    app.state.graph = GraphStateReader(
                        build_agent(llma, llmb, checkpointer=checkpoint.saver))
                    session_lock = SessionLock(redis_client, settings.redis_key_prefix)
                    app.state.session_lock = session_lock
                    async def invoke_graph(input_state, *, config, context, request_id=None):
                        return await run_owned_graph((llma, llmb), checkpoint.saver.serde,
                            settings.database_url, session_lock, input_state,
                            config=config, context=context, request_id=request_id)
                    app.state.invoke_graph = invoke_graph
            yield
        finally:
            if checkpoint is not None:
                await checkpoint.close()
            if redis_client is not None:
                await close_redis(redis_client)
            if engine is not None:
                await engine.dispose()

    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.include_router(router)
    app.include_router(chat_router)
    @app.exception_handler(ChatError)
    async def chat_error_handler(_request: Request, exc: ChatError):
        return JSONResponse(status_code=exc.status_code,
                            content={"error": exc.code, "message": exc.message})

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_request: Request, _exc: RequestValidationError):
        return JSONResponse(status_code=422,
                            content={"error": "invalid_request", "message": "Invalid request"})
    return app


app = create_app()
frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
