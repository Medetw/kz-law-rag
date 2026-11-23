"""Tokenisation for BM25: lowercase words and numbers, Russian Snowball stems, crude Kazakh stems.

Kazakh is agglutinative (салық -> салығының, салықтың) and has no Snowball stemmer, so Kazakh words are
cut to a fixed prefix after normalising the voiced/unvoiced ending (ғ/қ, г/к). Crude, but measurable:
see evals/README for recall on Kazakh questions.
"""

from __future__ import annotations

import re
from functools import lru_cache

import snowballstemmer

_WORD = re.compile(r"[0-9]+(?:[.,-][0-9]+)*|[a-zа-яёәғқңөұүһі]+", re.I)
_KZ_LETTERS = re.compile(r"[әғқңөұүһі]", re.I)
KZ_PREFIX = 6

STOP = frozenset(
    [
        "и",
        "в",
        "во",
        "на",
        "по",
        "с",
        "со",
        "к",
        "ко",
        "о",
        "об",
        "от",
        "до",
        "из",
        "за",
        "для",
        "не",
        "ни",
        "что",
        "как",
        "это",
        "или",
        "а",
        "но",
        "же",
        "ли",
        "бы",
        "то",
        "так",
        "при",
        "его",
        "ее",
        "их",
        "этот",
        "эта",
        "эти",
        "который",
        "которая",
        "которые",
        "также",
        "если",
        "быть",
        "является",
        "настоящего",
        "настоящей",
        "кодекса",
        "кодексом",
        "және",
        "мен",
        "бен",
        "пен",
        "да",
        "де",
        "та",
        "те",
        "үшін",
        "бойынша",
        "осы",
        "ол",
        "бұл",
        "немесе",
        "туралы",
        "сәйкес",
        "кодекстің",
        "кодексінің",
        "the",
        "of",
        "and",
        "to",
        "in",
        "for",
        "is",
        "on",
        "a",
        "an",
        "what",
        "how",
    ]
)


@lru_cache(maxsize=1)
def _ru_stemmer() -> snowballstemmer.stemmer:
    return snowballstemmer.stemmer("russian")


def is_kazakh(text: str) -> bool:
    return bool(_KZ_LETTERS.search(text))


@lru_cache(maxsize=200_000)
def _stem(word: str, kazakh: bool) -> str:
    if word[0].isdigit():
        return word
    if kazakh:
        w = word.replace("ғ", "қ").replace("г", "к")
        return w[:KZ_PREFIX]
    return str(_ru_stemmer().stemWord(word))


def tokenize(text: str) -> list[str]:
    kazakh = is_kazakh(text)
    return [_stem(w, kazakh) for w in (m.group(0).lower() for m in _WORD.finditer(text)) if w not in STOP]
