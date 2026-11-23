"""HTTP API: search and article lookup (no LLM needed), /ask (LangGraph agent), and a small demo page."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from typing import Annotated, Any

import openai
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from kzlaw import tools as T
from kzlaw.agent.graph import AnswerResult, answer_question
from kzlaw.agent.llm import MissingKeyError
from kzlaw.api import ui
from kzlaw.config import Settings, get_settings
from kzlaw.corpus import CODES
from kzlaw.search.hybrid import Retriever

log = logging.getLogger(__name__)

DISCLAIMER = (
    "Ответ сгенерирован ИИ по текстам кодексов с adilet.zan.kz и не является юридической консультацией. "
    "Проверяйте статьи по ссылкам."
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    retriever = getattr(app.state, "retriever", None)
    if retriever is None:
        retriever = Retriever(settings)
        app.state.retriever = retriever
    if not retriever.chunks:
        log.warning("index is empty: run `kzlaw ingest` (or `kzlaw load-vectors`) first")
    yield


def create_app(settings: Settings | None = None, retriever: Retriever | None = None) -> FastAPI:
    app = FastAPI(
        title="kz-law-rag",
        description="RAG over the Tax and Labour codes of Kazakhstan with article-level citations.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings or get_settings()
    app.state.retriever = retriever
    _routes(app)
    return app


def _retriever(request: Request) -> Retriever:
    r: Retriever = request.app.state.retriever
    if not r.chunks:
        raise HTTPException(503, "index is empty: run `kzlaw ingest` first")
    return r


def _settings(request: Request) -> Settings:
    s: Settings = request.app.state.settings
    return s


RetrieverDep = Annotated[Retriever, Depends(_retriever)]
SettingsDep = Annotated[Settings, Depends(_settings)]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    as_of: date | None = Field(None, description="date the question refers to; default today")


class AskResponse(BaseModel):
    answer: str
    citations: list[Any]
    invalid_citations: list[str]
    retrieved: list[str]
    tool_calls: list[str]
    as_of: str
    trace_id: str | None
    disclaimer: str = DISCLAIMER


def _ask(question: str, as_of: date | None, settings: Settings, retriever: Retriever) -> AnswerResult:
    try:
        return answer_question(question, as_of=as_of, settings=settings, retriever=retriever)
    except MissingKeyError as e:
        raise HTTPException(503, str(e)) from e
    except openai.APIError as e:  # provider down, timeout, rate limit: the client should retry later
        log.warning("LLM provider error: %r", e)
        raise HTTPException(502, f"LLM provider error ({type(e).__name__}); try again later") from e


def _routes(app: FastAPI) -> None:
    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        r: Retriever = request.app.state.retriever
        return {
            "status": "ok" if r.chunks else "empty_index",
            "chunks": len(r.chunks),
            "embedding_model": r.embedder.name,
            "codes": {c.id: c.title_ru for c in CODES.values()},
        }

    @app.get("/search", response_model=list[T.Source])
    def search(
        retriever: RetrieverDep,
        q: Annotated[str, Query(min_length=2, max_length=500)],
        k: Annotated[int, Query(ge=1, le=15)] = 6,
        as_of: date | None = None,
        code: str | None = None,
        lang: str | None = None,
    ) -> list[T.Source]:
        try:
            return T.search_law(retriever, q, as_of=as_of.isoformat() if as_of else None, code=code, lang=lang, k=k)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    @app.get("/articles/{code}/{article}", response_model=T.Source)
    def article(retriever: RetrieverDep, code: str, article: str, lang: str = "ru") -> T.Source:
        try:
            src = T.get_article(retriever, code, article, lang)
        except ValueError as e:
            raise HTTPException(404, str(e)) from e
        if src is None:
            raise HTTPException(404, f"article {code}:{article} not found")
        return src

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, retriever: RetrieverDep, settings: SettingsDep) -> AskResponse:
        res = _ask(body.question, body.as_of, settings, retriever)
        return AskResponse(**res.model_dump(exclude={"question", "contexts"}))

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request, q: str = "", mode: str = "ask") -> str:
        retriever: Retriever = request.app.state.retriever
        if not q.strip():
            return ui.page(q, mode)
        if not retriever.chunks:
            return ui.page(q, mode, error="Индекс пуст: сначала запустите kzlaw ingest.")
        if mode == "search":
            return ui.page(q, mode, sources=T.search_law(retriever, q, k=8))
        try:
            res = answer_question(q, settings=request.app.state.settings, retriever=retriever)
        except MissingKeyError as e:
            return ui.page(q, mode, error=f"{e}. Поиск без LLM работает в режиме «Только поиск».")
        except openai.APIError as e:
            log.warning("LLM provider error: %r", e)
            return ui.page(
                q,
                mode,
                error="LLM сейчас недоступна, попробуйте позже. Поиск без LLM работает в режиме «Только поиск».",
            )
        return ui.page(q, mode, result=res, disclaimer=DISCLAIMER)


app = create_app()
