"""Which legal acts are in the corpus and when each one is in force.

Every citation in an answer uses the form ``code:article`` (for example ``tax2026:503``),
so the code ids below are part of the public contract of the API and the MCP server.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

Lang = Literal["ru", "kz"]
ADILET_LANG: dict[Lang, str] = {"ru": "rus", "kz": "kaz"}


@dataclass(frozen=True)
class Code:
    id: str
    ngr: str  # registration number on adilet.zan.kz
    title_ru: str
    title_kz: str
    short_ru: str
    valid_from: date
    valid_to: date | None
    langs: tuple[Lang, ...]

    def in_force(self, on: date) -> bool:
        return self.valid_from <= on and (self.valid_to is None or on <= self.valid_to)

    def url(self, lang: Lang, anchor: str | None = None) -> str:
        base = f"https://adilet.zan.kz/{ADILET_LANG[lang]}/docs/{self.ngr}"
        return f"{base}#{anchor}" if anchor else base


CODES: dict[str, Code] = {
    c.id: c
    for c in (
        Code(
            id="tax2026",
            ngr="K2500000214",
            title_ru="Налоговый кодекс РК (редакция с 2026 года)",
            title_kz="ҚР Салық кодексі (2026 жылдан бастап)",
            short_ru="НК РК 2026",
            valid_from=date(2026, 1, 1),
            valid_to=None,
            langs=("ru", "kz"),
        ),
        Code(
            id="tax2017",
            ngr="K1700000120",
            title_ru="Налоговый кодекс РК 2017 года (действовал 2018-2025)",
            title_kz="ҚР Салық кодексі 2017 жыл (2018-2025 жылдары қолданылды)",
            short_ru="НК РК 2017",
            valid_from=date(2018, 1, 1),
            valid_to=date(2025, 12, 31),
            langs=("ru",),
        ),
        Code(
            id="labor",
            ngr="K1500000414",
            title_ru="Трудовой кодекс РК",
            title_kz="ҚР Еңбек кодексі",
            short_ru="ТК РК",
            valid_from=date(2016, 1, 1),
            valid_to=None,
            langs=("ru", "kz"),
        ),
    )
}

TAX_CODES = ("tax2026", "tax2017")


def codes_in_force(on: date) -> list[str]:
    return [c.id for c in CODES.values() if c.in_force(on)]


def cite_label(code_id: str, article: str) -> str:
    """Human label for a citation: ``НК РК 2026, ст. 503``."""
    return f"{CODES[code_id].short_ru}, ст. {article}"
