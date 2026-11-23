"""PostgreSQL + pgvector storage for chunks."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from kzlaw.corpus import Lang
from kzlaw.embeddings import Vector
from kzlaw.ingest.parse import Chunk


@dataclass(frozen=True)
class StoredChunk:
    id: str
    code: str
    article: str
    lang: Lang
    title: str
    heading: str
    text: str
    points: list[str]
    url: str

    @property
    def cite_id(self) -> str:
        return f"{self.code}:{self.article}"

    @property
    def full_text(self) -> str:
        return f"{self.heading}\n{self.text}"


_COLS = "id, code, article, lang, title, heading, text, points, url"


def connect(url: str) -> psycopg.Connection[dict[str, object]]:
    conn = psycopg.connect(url, row_factory=dict_row, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def init_schema(conn: psycopg.Connection[dict[str, object]], model: str, dim: int, reset: bool = False) -> None:
    if reset:
        conn.execute("DROP TABLE IF EXISTS chunks; DROP TABLE IF EXISTS meta")
    conn.execute("CREATE TABLE IF NOT EXISTS meta (key text PRIMARY KEY, value text NOT NULL)")
    row = conn.execute("SELECT value FROM meta WHERE key = 'embedding_model'").fetchone()
    if row and row["value"] != f"{model}:{dim}":
        raise RuntimeError(f"DB was built with {row['value']}, settings say {model}:{dim}. Re-run ingest with --reset.")
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id text PRIMARY KEY,
            code text NOT NULL,
            article text NOT NULL,
            lang text NOT NULL,
            part int NOT NULL,
            title text NOT NULL,
            heading text NOT NULL,
            text text NOT NULL,
            points text[] NOT NULL,
            url text NOT NULL,
            embedding vector({dim}) NOT NULL
        )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)")
    conn.execute("CREATE INDEX IF NOT EXISTS chunks_code_article ON chunks (code, article)")
    conn.execute("INSERT INTO meta VALUES ('embedding_model', %s) ON CONFLICT (key) DO NOTHING", (f"{model}:{dim}",))


def existing_ids(conn: psycopg.Connection[dict[str, object]]) -> set[str]:
    return {str(r["id"]) for r in conn.execute("SELECT id FROM chunks").fetchall()}


def upsert(conn: psycopg.Connection[dict[str, object]], chunks: Sequence[Chunk], vectors: Vector) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO chunks (id, code, article, lang, part, title, heading, text, points, url, embedding)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (id) DO UPDATE SET text = EXCLUDED.text, heading = EXCLUDED.heading,
                   points = EXCLUDED.points, url = EXCLUDED.url, embedding = EXCLUDED.embedding""",
            [
                (c.id, c.code, c.article, c.lang, c.part, c.title, c.heading, c.text, list(c.points), c.url, v)
                for c, v in zip(chunks, vectors, strict=True)
            ],
        )


def _row(r: dict[str, object]) -> StoredChunk:
    return StoredChunk(**{k: r[k] for k in _COLS.split(", ")})  # type: ignore[arg-type]


def all_chunks(conn: psycopg.Connection[dict[str, object]]) -> list[StoredChunk]:
    """All stored chunks; empty list if nothing was ingested yet."""
    if conn.execute("SELECT to_regclass('chunks') AS t").fetchone() == {"t": None}:
        return []
    rows = conn.execute(f"SELECT {_COLS} FROM chunks ORDER BY id").fetchall()  # noqa: S608 - constant column list
    return [_row(r) for r in rows]


def embedding_model(conn: psycopg.Connection[dict[str, object]]) -> str | None:
    if conn.execute("SELECT to_regclass('meta') AS t").fetchone() == {"t": None}:
        return None
    row = conn.execute("SELECT value FROM meta WHERE key = 'embedding_model'").fetchone()
    return str(row["value"]) if row else None


def vector_search(
    conn: psycopg.Connection[dict[str, object]],
    query_vec: Vector,
    codes: Iterable[str],
    langs: Iterable[str],
    limit: int,
) -> list[tuple[StoredChunk, float]]:
    conn.execute("SET hnsw.ef_search = 200")
    conn.execute("SET hnsw.iterative_scan = relaxed_order")  # keep returning rows when filters drop many
    rows = conn.execute(
        f"""SELECT {_COLS}, 1 - (embedding <=> %s) AS score FROM chunks
            WHERE code = ANY(%s) AND lang = ANY(%s)
            ORDER BY embedding <=> %s LIMIT %s""",  # noqa: S608 - constant column list
        (np.asarray(query_vec), list(codes), list(langs), np.asarray(query_vec), limit),
    ).fetchall()
    return [(_row(r), float(r["score"])) for r in rows]  # type: ignore[arg-type]


def get_article(conn: psycopg.Connection[dict[str, object]], code: str, article: str, lang: str) -> list[StoredChunk]:
    rows = conn.execute(
        f"SELECT {_COLS} FROM chunks WHERE code = %s AND article = %s AND lang = %s ORDER BY part",  # noqa: S608
        (code, article, lang),
    ).fetchall()
    return [_row(r) for r in rows]
