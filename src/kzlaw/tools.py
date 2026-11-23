"""Domain tools shared by the LangGraph agent and the MCP server."""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel

from kzlaw.corpus import CODES, cite_label
from kzlaw.search.hybrid import Retriever

ARTICLE_CHAR_LIMIT = 8000
MERGE_CHUNKS = 2  # at most this many chunks of one article in a search result


class Source(BaseModel):
    cite_id: str  # tax2026:503 — use this in answers
    label: str  # НК РК 2026, ст. 503
    title: str
    lang: str
    points: list[str]
    url: str
    text: str


def _parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def search_law(
    retriever: Retriever,
    query: str,
    as_of: str | None = None,
    code: str | None = None,
    lang: str | None = None,
    k: int = 6,
) -> list[Source]:
    """Hybrid search over the codes in force on ``as_of`` (YYYY-MM-DD, default today).

    Returns up to ``k`` distinct articles; several matching chunks of one article are merged (best first).
    """
    if code is not None and code not in CODES:
        raise ValueError(f"unknown code {code!r}; known: {sorted(CODES)}")
    if lang is not None and lang not in ("ru", "kz"):
        raise ValueError(f"unknown lang {lang!r}; use ru or kz")
    k = max(1, min(k, 15))
    hits = retriever.search(
        query,
        k=k * 3,
        codes=[code] if code else None,
        langs=[lang] if lang else None,
        as_of=_parse_date(as_of),
    )
    grouped: dict[tuple[str, str], list[Any]] = {}
    for h in hits:
        key = (h.chunk.cite_id, h.chunk.lang)
        if key in grouped or len(grouped) < k:
            grouped.setdefault(key, []).append(h.chunk)
    out = []
    for chunks in grouped.values():
        first = chunks[0]
        points = list(dict.fromkeys(p for c in chunks[:MERGE_CHUNKS] for p in c.points))
        out.append(
            Source(
                cite_id=first.cite_id,
                label=cite_label(first.code, first.article),
                title=first.title,
                lang=first.lang,
                points=points,
                url=first.url,
                text="\n…\n".join(c.text for c in chunks[:MERGE_CHUNKS]),
            )
        )
    return out


def get_article(retriever: Retriever, code: str, article: str, lang: str = "ru") -> Source | None:
    """Full text of one article, e.g. code="tax2026", article="503"."""
    if code not in CODES:
        raise ValueError(f"unknown code {code!r}; known: {sorted(CODES)}")
    parts = retriever.article(code, article, lang) or retriever.article(code, article, "ru")
    if not parts:
        return None
    first = parts[0]
    text = "\n".join(p.text for p in parts)
    if len(text) > ARTICLE_CHAR_LIMIT:
        text = text[:ARTICLE_CHAR_LIMIT] + " …[обрезано]"
    points = list(dict.fromkeys(pt for p in parts for pt in p.points))
    return Source(
        cite_id=first.cite_id,
        label=cite_label(code, article),
        title=first.title,
        lang=first.lang,
        points=points,
        url=first.url,
        text=text,
    )


class EditionsComparison(BaseModel):
    tax2017: list[Source]  # Tax Code 2017 (in force 2018-2025)
    tax2026: list[Source]  # Tax Code 2025 (in force from 2026)


def compare_editions(retriever: Retriever, query: str, k: int = 3) -> EditionsComparison:
    """How the same topic is regulated in the old (2018-2025) and the new (2026+) Tax Code."""
    return EditionsComparison(
        tax2017=search_law(retriever, query, code="tax2017", lang="ru", k=k),
        tax2026=search_law(retriever, query, code="tax2026", lang="ru", k=k),
    )


def format_sources(sources: list[Source]) -> str:
    """Compact text for the LLM: one block per article, chunks of the same article merged."""
    if not sources:
        return "Ничего не найдено."
    merged: dict[str, list[Source]] = {}
    for s in sources:
        merged.setdefault(s.cite_id, []).append(s)
    blocks = []
    for cite_id, group in merged.items():
        s = group[0]
        points = sorted({p for g in group for p in g.points}, key=lambda p: [int(x) for x in p.split("-")])
        pts = f" (пп. {', '.join(points)})" if points else ""
        body = "\n…\n".join(g.text for g in group)
        blocks.append(f"[{cite_id}] {s.label} «{s.title}»{pts} [{s.lang}]\n{body}")
    return "\n\n---\n\n".join(blocks)
