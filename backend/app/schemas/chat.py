from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str = Field(min_length=1, max_length=128)


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    service_status: str
    tool_call_count: int
