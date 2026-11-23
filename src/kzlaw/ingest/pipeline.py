"""adilet HTML -> articles -> chunks -> embeddings -> pgvector. Resumable: already stored chunks are skipped."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from kzlaw import db
from kzlaw.config import Settings
from kzlaw.corpus import CODES
from kzlaw.embeddings import get_embedder
from kzlaw.ingest.adilet import fetch_html
from kzlaw.ingest.parse import Chunk, chunk_article, parse_articles

log = logging.getLogger(__name__)
BATCH = 32


def build_chunks(settings: Settings, code_ids: list[str] | None = None) -> list[Chunk]:
    chunks: list[Chunk] = []
    for cid in code_ids or list(CODES):
        code = CODES[cid]
        for lang in code.langs:
            articles = parse_articles(fetch_html(code, lang, settings.raw_dir), cid, lang)
            new = [c for a in articles for c in chunk_article(a)]
            log.info("%s/%s: %d articles, %d chunks", cid, lang, len(articles), len(new))
            chunks.extend(new)
    return chunks


def export_chunks(settings: Settings, path: Path, code_ids: list[str] | None = None) -> int:
    """Write chunks as JSONL (id + text to embed) for bulk embedding elsewhere (scripts/embed_gpu.py)."""
    chunks = build_chunks(settings, code_ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"id": c.id, "text": c.full_text}, ensure_ascii=False) + "\n")
    return len(chunks)


def load_vectors(settings: Settings, vectors_path: Path, reset: bool = False) -> int:
    """Store chunks with vectors computed by scripts/embed_gpu.py. The model must match the query embedder."""
    data = np.load(vectors_path)
    model = str(data["model"])
    vectors = {str(i): v for i, v in zip(data["ids"], data["vectors"], strict=True)}
    chunks = [c for c in build_chunks(settings) if c.id in vectors]
    missing = len(vectors) - len(chunks)
    if missing:
        raise RuntimeError(f"{missing} vector ids have no matching chunk: re-export chunks and re-embed")
    dim = int(data["vectors"].shape[1])
    with db.connect(settings.database_url) as conn:
        db.init_schema(conn, model, dim, reset=reset)
        for i in range(0, len(chunks), 256):
            batch = chunks[i : i + 256]
            db.upsert(conn, batch, np.stack([vectors[c.id] for c in batch]))
    log.info("stored %d chunks with %s vectors (dim %d)", len(chunks), model, dim)
    return len(chunks)


def ingest(settings: Settings, code_ids: list[str] | None = None, reset: bool = False) -> int:
    chunks = build_chunks(settings, code_ids)
    embedder = get_embedder(settings)
    with db.connect(settings.database_url) as conn:
        db.init_schema(conn, embedder.name, embedder.dim, reset=reset)
        done = db.existing_ids(conn)
        todo = [c for c in chunks if c.id not in done]
        log.info(
            "%d chunks total, %d already stored, %d to embed with %s", len(chunks), len(done), len(todo), embedder.name
        )
        t0 = time.monotonic()
        for i in range(0, len(todo), BATCH):
            batch = todo[i : i + BATCH]
            db.upsert(conn, batch, embedder.embed_documents([c.full_text for c in batch]))
            n = i + len(batch)
            rate = n / max(time.monotonic() - t0, 1e-6)
            log.info("embedded %d/%d (%.1f chunks/s, ~%.0f min left)", n, len(todo), rate, (len(todo) - n) / rate / 60)
    return len(todo)
