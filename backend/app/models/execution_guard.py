"""Durable execution ownership for a LangGraph thread."""

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class ExecutionGuard(Base):
    __tablename__ = "execution_guards"
    thread_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    execution_epoch: Mapped[int] = mapped_column(BigInteger, nullable=False)
    request_id: Mapped[str] = mapped_column(String(128), nullable=False)
    owner_token: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
