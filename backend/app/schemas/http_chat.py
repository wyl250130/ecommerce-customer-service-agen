from uuid import UUID

from pydantic import BaseModel, Field


class CreateSessionResponse(BaseModel):
    id: UUID
    service_status: str


class SessionResponse(CreateSessionResponse):
    created_at: str


class SendMessageBody(BaseModel):
    content: str = Field(min_length=1, max_length=4000)
    idempotency_key: str = Field(min_length=1, max_length=128)


class ChatRequestResponse(BaseModel):
    id: UUID
    session_id: UUID
    execution_status: str
    result_type: str | None
    response: str | None
    error_code: str | None

    @classmethod
    def from_row(cls, row):
        return cls(id=row.id, session_id=row.session_id,
                   execution_status=row.execution_status.value,
                   result_type=row.result_type, response=row.response,
                   error_code=row.error_code)


class MessageResponse(BaseModel):
    id: UUID
    request_id: UUID | None
    sender_type: str
    content: str
    sequence_no: int

    @classmethod
    def from_row(cls, row):
        return cls(id=row.id, request_id=row.request_id,
                   sender_type=row.sender_type.value, content=row.content,
                   sequence_no=row.sequence_no)


class ErrorResponse(BaseModel):
    error: str
    message: str
