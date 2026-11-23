"""Turn the HTML of a code from adilet.zan.kz into articles and retrieval chunks.

The page structure (checked on the Tax Code 2026 and the Labour Code):

* headings ``<h3>``: part, section (РАЗДЕЛ / БӨЛІМ), chapter (Глава / тарау), paragraph;
* an article starts with a block whose text is ``Статья 503. Title`` (RU, inside ``<p><b>``)
  or ``503-бап. Title`` (KZ, inside ``<h3>``); the block carries an anchor ``z…`` used for deep links;
* editorial notes are ``<span class="note">`` (Сноска) and red ``<font>`` (Примечание ИЗПИ) — dropped;
* tables (rates, asset groups) are flattened to ``cell | cell`` rows.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field

from kzlaw.corpus import CODES, Lang

_NOTE = re.compile(r'<span class="note">.*?</span>|<font color="#?FF0000">.*?</font>', re.S | re.I)
_TABLE = re.compile(r"<table.*?</table>", re.S | re.I)
_BLOCK = re.compile(r"<(h3|p)\b([^>]*)>(.*?)</\1>", re.S | re.I)
_ANCHOR = re.compile(r'(?:id|name)="(z\d+)"')
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_ART_RU = re.compile(r"^Статья\s+(\d+(?:-\d+)?)\.\s*(.*)$")
_ART_KZ = re.compile(r"^(\d+(?:-\d+)?)-бап\.\s*(.*)$")
_HEADING = re.compile(r"^(РАЗДЕЛ|Глава|Параграф|\d+-БӨЛІМ|\d+-тарау|\d+-параграф|ОБЩАЯ|ОСОБЕННАЯ|ЖАЛПЫ|ЕРЕКШЕ)", re.I)
_POINT = re.compile(r"^(\d+(?:-\d+)?)\.\s")

MAX_CHARS = 1500  # ~400 tokens for XLM-R tokenizers; e5 models accept 512


@dataclass
class Article:
    code: str
    lang: Lang
    number: str
    title: str
    anchor: str | None
    path: list[str]  # section / chapter headings above the article
    paragraphs: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.paragraphs)


@dataclass(frozen=True)
class Chunk:
    id: str  # tax2026:503:ru:0
    code: str
    article: str
    lang: Lang
    part: int
    title: str
    heading: str  # contextual header prepended to the text before embedding / BM25
    text: str
    points: tuple[str, ...]  # пункты covered by this chunk, e.g. ("1", "2")
    url: str

    @property
    def cite_id(self) -> str:
        return f"{self.code}:{self.article}"

    @property
    def full_text(self) -> str:
        return f"{self.heading}\n{self.text}"


def clean(fragment: str) -> str:
    text = html.unescape(_TAG.sub(" ", fragment)).replace("\xa0", " ")
    return _WS.sub(" ", text).strip()


def _flatten_table(match: re.Match[str]) -> str:
    rows = []
    for row in re.findall(r"<tr.*?</tr>", match.group(0), re.S | re.I):
        cells = [clean(c) for c in re.findall(r"<td.*?</td>", row, re.S | re.I)]
        if any(cells):
            rows.append(" | ".join(cells))
    return "<p>" + html.escape(" ; ".join(rows)) + "</p>"


def parse_articles(raw_html: str, code: str, lang: Lang) -> list[Article]:
    body = _NOTE.sub(" ", raw_html)
    body = _TABLE.sub(_flatten_table, body)
    art_re = _ART_RU if lang == "ru" else _ART_KZ
    articles: list[Article] = []
    path: list[str] = []
    current: Article | None = None
    for tag, attrs, inner in _BLOCK.findall(body):
        text = clean(inner)
        if not text:
            continue
        m = art_re.match(text)
        if m:
            anchor = _ANCHOR.search(attrs) or _ANCHOR.search(inner)
            current = Article(
                code, lang, m.group(1), m.group(2).strip(), anchor.group(1) if anchor else None, list(path)
            )
            articles.append(current)
            continue
        if tag.lower() == "h3" and _HEADING.match(text):
            level = _heading_level(text)
            path = [*path[:level], text]
            current = None
            continue
        if current is not None:
            current.paragraphs.append(text)
    # adilet sometimes repeats an article (e.g. a version table of contents); keep the longest copy
    best: dict[str, Article] = {}
    for a in articles:
        if a.number not in best or len(a.text) > len(best[a.number].text):
            best[a.number] = a
    return list(best.values())


def _heading_level(text: str) -> int:
    t = text.lower()
    if t.startswith(("общая", "особенная", "жалпы", "ерекше")):
        return 0
    if t.startswith("раздел") or "бөлім" in t.split(".")[0]:
        return 1
    if t.startswith("глава") or "тарау" in t.split(".")[0]:
        return 2
    return 3


def chunk_article(a: Article) -> list[Chunk]:
    """Split an article into chunks of whole points (пункты); never cut mid-sentence unless a point is huge."""
    code = CODES[a.code]
    label = f"Статья {a.number}" if a.lang == "ru" else f"{a.number}-бап"
    title_code = code.title_ru if a.lang == "ru" else code.title_kz
    heading = f"{title_code}. {label}. {a.title}"
    pieces: list[tuple[str | None, str]] = []
    point: str | None = None
    for p in a.paragraphs or [a.title]:
        m = _POINT.match(p)
        if m:
            point = m.group(1)
        for part in _split_long(p):
            pieces.append((point, part))

    groups: list[list[tuple[str | None, str]]] = [[]]
    size = 0
    for item in pieces:
        if groups[-1] and size + len(item[1]) > MAX_CHARS:
            groups.append([])
            size = 0
        groups[-1].append(item)
        size += len(item[1]) + 1

    chunks = []
    for i, g in enumerate(groups):
        points = tuple(dict.fromkeys(pt for pt, _ in g if pt))
        chunks.append(
            Chunk(
                id=f"{a.code}:{a.number}:{a.lang}:{i}",
                code=a.code,
                article=a.number,
                lang=a.lang,
                part=i,
                title=a.title,
                heading=heading,
                text="\n".join(t for _, t in g),
                points=points,
                url=code.url(a.lang, a.anchor),
            )
        )
    return chunks


def _split_long(text: str) -> list[str]:
    if len(text) <= MAX_CHARS:
        return [text]
    out: list[str] = []
    buf = ""
    for sentence in re.split(r"(?<=[.;:])\s+", text):
        if buf and len(buf) + len(sentence) > MAX_CHARS:
            out.append(buf)
            buf = ""
        while len(sentence) > MAX_CHARS:  # a single giant sentence (long lists in tables)
            out.append(sentence[:MAX_CHARS])
            sentence = sentence[MAX_CHARS:]
        buf = f"{buf} {sentence}".strip()
    if buf:
        out.append(buf)
    return out
