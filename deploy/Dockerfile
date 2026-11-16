FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock README.md /app/
COPY agent /app/agent
COPY rag /app/rag
COPY backend /app/backend
COPY frontend /app/frontend
COPY data /app/data
RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
