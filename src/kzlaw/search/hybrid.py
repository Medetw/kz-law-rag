"""Hybrid retrieval: pgvector (meaning, cross-lingual) + BM25 (exact words and numbers), fused with RRF,
optionally reranked by a cross-encoder. Filters: which codes were in force on a date, which languages."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np

from kzlaw import db
from kzlaw.config import Settings
from kzlaw.corpus import CODES, codes_in_force
from kzlaw.embeddings import Embedder, get_embedder
from kzlaw.search.bm25 import BM25
from kzlaw.search.expand import expand as expand_query
from kzlaw.search.expand import strip_years

log = logging.getLogger(__name__)
Mode = Literal["hybrid", "vector", "bm25"]
RRF_K = 60
CANDIDATES = 50
TITLE_BOOST = 2  # article title counted this many extra times in BM25 (a cheap BM25F)


@dataclass(frozen=True)
class Hit:
    chunk: db.StoredChunk
    score: float


def rrf(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank). Scales of the inputs don't matter."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


class Retriever:
    def __init__(self, settings: Settings, embedder: Embedder | None = None) -> None:
        self.settings = settings
        self.embedder = embedder or get_embedder(settings)
        self.conn = db.connect(settings.database_url)
        self.chunks = db.all_chunks(self.conn)
        self.by_id = {c.id: i for i, c in enumerate(self.chunks)}
        self.bm25 = BM25([f"{c.heading} {' '.join([c.title] * TITLE_BOOST)}\n{c.text}" for c in self.chunks])
        self._codes = np.array([c.code for c in self.chunks])
        self._langs = np.array([c.lang for c in self.chunks])
        self._reranker: object | None = None
        log.info("retriever ready: %d chunks", len(self.chunks))

    def resolve_codes(self, codes: list[str] | None, as_of: date | None) -> list[str]:
        if codes:
            unknown = set(codes) - set(CODES)
            if unknown:
                raise ValueError(f"unknown code(s): {sorted(unknown)}; known: {sorted(CODES)}")
            return codes
        return codes_in_force(as_of or date.today())  # noqa: DTZ011 - legal dates are calendar dates in KZ

    def search(
        self,
        query: str,
        k: int = 8,
        mode: Mode = "hybrid",
        rerank: bool | None = None,
        expand: bool = True,
        codes: list[str] | None = None,
        langs: list[str] | None = None,
        as_of: date | None = None,
    ) -> list[Hit]:
        code_ids = self.resolve_codes(codes, as_of)
        lang_ids = langs or ["ru", "kz"]
        original = query
        query = strip_years(query)  # the date is a filter (as_of), not a keyword
        if expand:
            query = expand_query(query)
        rankings: list[list[str]] = []
        scores: dict[str, float] = {}
        if mode in ("hybrid", "vector"):
            vec = db.vector_search(self.conn, self.embedder.embed_query(query), code_ids, lang_ids, CANDIDATES)
            rankings.append([c.id for c, _ in vec])
            scores.update({c.id: s for c, s in vec})
        if mode in ("hybrid", "bm25"):
            mask = np.isin(self._codes, code_ids) & np.isin(self._langs, lang_ids)
            lex = self.bm25.top(query, CANDIDATES, mask)
            rankings.append([self.chunks[i].id for i, _ in lex])
            if mode == "bm25":
                scores.update({self.chunks[i].id: s for i, s in lex})
        if mode == "hybrid":
            scores = rrf(rankings)
        ordered = sorted(scores, key=lambda cid: -scores[cid])
        hits = [Hit(self.chunks[self.by_id[cid]], scores[cid]) for cid in ordered if cid in self.by_id]
        if rerank if rerank is not None else self.settings.rerank:
            hits = self._rerank(original, hits[:30])
        return hits[:k]

    def _rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        from kzlaw.search.rerank import CrossEncoderReranker

        if self._reranker is None:
            self._reranker = CrossEncoderReranker(self.settings.reranker_model)
        if not isinstance(self._reranker, CrossEncoderReranker):
            raise TypeError("reranker not initialised")
        new_scores = self._reranker.score(query, [h.chunk.full_text for h in hits])
        return [Hit(h.chunk, s) for h, s in sorted(zip(hits, new_scores, strict=True), key=lambda x: -x[1])]

    def article(self, code: str, article: str, lang: str = "ru") -> list[db.StoredChunk]:
        return db.get_article(self.conn, code, article, lang)


_RETRIEVERS: dict[tuple[str, str, str], Retriever] = {}


def get_retriever(settings: Settings) -> Retriever:
    """One retriever per (database, embedder): loading chunks + BM25 + the model takes seconds."""
    key = (settings.database_url, settings.embedding_backend, settings.embedding_model)
    if key not in _RETRIEVERS:
        _RETRIEVERS[key] = Retriever(settings)
    return _RETRIEVERS[key]
