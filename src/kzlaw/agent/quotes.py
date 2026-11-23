"""Verbatim quotes: every «quoted» fragment of an answer must appear word for word in a retrieved article.

Legal meaning lives in small words ("по истечении" vs "в течение"), so conditions and deadlines are quoted, and
the quote is checked against the text the tools returned. Only typography is forgiven: case, ё/е, dash and quote
characters, whitespace. A quote may skip text with an ellipsis; each part must then be found, in order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MIN_WORDS = 4  # shorter «...» are names and terms («НДС», Закон «О гражданстве»), not quotes

_QUOTE = re.compile(r"«([^«»]+)»|“([^“”]+)”|\"([^\"]+)\"")
_ELLIPSIS = re.compile(r"\s*(?:…|\.\.\.|\[…\]|\(…\))\s*")
_DASHES = str.maketrans({"–": "-", "—": "-", "‑": "-", "−": "-", "ё": "е", "Ё": "Е", "\xa0": " "})
_EDGE_PUNCT = " .,;:!?()[]"


@dataclass(frozen=True)
class QuoteCheck:
    verified: list[str]
    invalid: list[str]


def normalize(text: str) -> str:
    text = text.translate(_DASHES).lower()
    text = re.sub(r"[«»“”\"']", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r" (?=[,.;:)])|(?<=\() ", "", text)  # "1) , 2)" in the source vs "1), 2)" in the quote
    return text.strip(_EDGE_PUNCT)


def extract(answer: str) -> list[str]:
    """Quoted fragments of at least MIN_WORDS words, in order of appearance."""
    out = []
    for m in _QUOTE.finditer(answer):
        q = next(g for g in m.groups() if g is not None).strip()
        if len(q.split()) >= MIN_WORDS:
            out.append(q)
    return out


def is_verbatim(quote: str, haystack: str) -> bool:
    """True if every ellipsis-separated part of the quote occurs in the haystack, in order."""
    pos = 0
    for part in (normalize(p) for p in _ELLIPSIS.split(quote)):
        if not part:
            continue
        found = haystack.find(part, pos)
        if found < 0:
            return False
        pos = found + len(part)
    return True


def check(answer: str, sources: list[str]) -> QuoteCheck:
    haystacks = [normalize(s) for s in sources]
    verified: list[str] = []
    invalid: list[str] = []
    for q in extract(answer):
        (verified if any(is_verbatim(q, h) for h in haystacks) else invalid).append(q)
    return QuoteCheck(verified, invalid)


def unquote(answer: str, invalid: list[str]) -> str:
    """Drop the quote marks around quotes that are not verbatim: the text stays, the claim of a quote goes."""
    for q in invalid:
        for left, right in (("«", "»"), ("“", "”"), ('"', '"')):
            answer = answer.replace(f"{left}{q}{right}", q)
    return answer
