"""Shared SQLAlchemy metadata for future ORM models; no business tables yet."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
