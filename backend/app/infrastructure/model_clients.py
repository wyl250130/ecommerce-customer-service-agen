"""Create the two long-lived, independently configured chat models."""

from langchain_openai import ChatOpenAI

from .app_config import AppSettings


def create_model_clients(settings: AppSettings):
    missing = [name for name, value in {
        "CS_LLM_API_KEY": settings.llm_api_key,
        "CS_LLMA_MODEL": settings.llma_model,
        "CS_LLMB_MODEL": settings.llmb_model,
    }.items() if not value]
    if missing:
        raise ValueError("Missing model configuration: " + ", ".join(missing))
    common = {"api_key": settings.llm_api_key, "timeout": settings.llm_timeout}
    if settings.llm_base_url:
        common["base_url"] = settings.llm_base_url
    return (ChatOpenAI(model=settings.llma_model, **common),
            ChatOpenAI(model=settings.llmb_model, **common))
