"""Integration test against a real PostgreSQL + pgvector.

Runs only when TEST_DATABASE_URL is set, and drops/recreates the tables in that database, so point it at a
throwaway database (CI uses a service container), never at the one with your real index.
"""

import os

import pytest

from kzlaw import db
from kzlaw.config import Settings
from kzlaw.embeddings import HashEmbedder
from kzlaw.ingest.parse import chunk_article, parse_articles
from kzlaw.search.hybrid import Retriever

from .conftest import TAX_KZ_HTML, TAX_RU_HTML

URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = [pytest.mark.db, pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")]


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    assert URL
    emb = HashEmbedder()
    chunks = [
        c
        for html, lang in ((TAX_RU_HTML, "ru"), (TAX_KZ_HTML, "kz"))
        for a in parse_articles(html, "tax2026", lang)
        for c in chunk_article(a)
    ]  # type: ignore[arg-type]
    with db.connect(URL) as conn:
        db.init_schema(conn, emb.name, emb.dim, reset=True)
        db.upsert(conn, chunks, emb.embed_documents([c.full_text for c in chunks]))
        db.upsert(conn, chunks[:1], emb.embed_documents([chunks[0].full_text]))  # idempotent re-run
    settings = Settings(_env_file=None, database_url=URL, embedding_backend="hash")  # type: ignore[call-arg]
    return Retriever(settings, embedder=emb)


def test_rows_stored_once(retriever: Retriever) -> None:
    assert len(retriever.chunks) == 4
    assert db.embedding_model(retriever.conn) == "hash-256:256"


@pytest.mark.parametrize("mode", ["vector", "bm25", "hybrid"])
def test_each_mode_finds_the_period_article(retriever: Retriever, mode: str) -> None:
    hits = retriever.search("налоговый период по НДС", k=3, mode=mode, codes=["tax2026"])  # type: ignore[arg-type]
    assert hits[0].chunk.cite_id == "tax2026:504"


def test_language_filter(retriever: Retriever) -> None:
    hits = retriever.search("ставка", k=5, langs=["kz"], codes=["tax2026"])
    assert {h.chunk.lang for h in hits} == {"kz"}


def test_article_lookup_and_model_guard(retriever: Retriever) -> None:
    assert retriever.article("tax2026", "503", "kz")[0].title == "Қосылған құн салығының мөлшерлемелері"
    with pytest.raises(RuntimeError, match="Re-run ingest"):
        db.init_schema(retriever.conn, "another-model", 128)


def test_get_retriever_is_cached_per_database() -> None:
    from kzlaw.search.hybrid import get_retriever

    assert URL
    s = Settings(_env_file=None, database_url=URL, embedding_backend="hash")  # type: ignore[call-arg]
    assert get_retriever(s) is get_retriever(s.model_copy())
