FROM python:3.12-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /app

# dependencies first (cached layer); no torch: embeddings run on onnxruntime
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-default-groups --no-install-project

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-default-groups

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "kzlaw.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
