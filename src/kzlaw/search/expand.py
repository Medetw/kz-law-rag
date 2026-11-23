"""Query expansion: abbreviations and everyday words -> the wording the codes actually use.

The codes almost never say "НДС" or "декрет"; they say "налог на добавленную стоимость" and
"отпуск по беременности и родам". Neither BM25 nor a small embedding model bridges that reliably,
so the query gets the legal wording appended. The list is domain knowledge, not tuned per question.
"""

from __future__ import annotations

import re

_TERMS: dict[str, str] = {
    # RU abbreviations
    r"\bндс\b": "налог на добавленную стоимость",
    r"\bкпн\b": "корпоративный подоходный налог",
    r"\bипн\b": "индивидуальный подоходный налог",
    r"\bмрп\b": "месячный расчетный показатель",
    r"\bип\b": "индивидуальный предприниматель",
    r"\bтоо\b": "товарищество с ограниченной ответственностью",
    r"\bэсф\b": "электронный счет-фактура",
    r"\bопв\b": "обязательные пенсионные взносы",
    r"\bосмс\b": "обязательное социальное медицинское страхование",
    r"\bкфх\b": "крестьянское или фермерское хозяйство",
    r"\bнк\b": "налоговый кодекс",
    r"\bтк\b": "трудовой кодекс",
    # RU everyday words
    r"\bдекрет\w*": "отпуск по беременности и родам",
    r"\bбольничн\w*": "временная нетрудоспособность",
    r"\bзарплат\w*|\bзп\b": "заработная плата",
    r"увол\w* по собственному желанию|\bувол\w*": "расторжение трудового договора по инициативе работника",
    r"\bудаленк\w*|\bудаленн\w* работ\w*": "дистанционная работа",
    r"\bупрощенк\w*": "специальный налоговый режим на основе упрощенной декларации",
    r"\bпереработк\w*": "сверхурочная работа",
    # KZ abbreviations
    r"\bққс\w*": "қосылған құн салығы",
    r"\bжтс\w*": "жеке табыс салығы",
    r"\bктс\w*": "корпоративтік табыс салығы",
    r"\bаек\w*": "айлық есептік көрсеткіш",
    r"\bжк\b": "жеке кәсіпкер",
    # EN abbreviations only (full translation is the agent's job: it rewrites queries into legal Russian)
    r"\bvat\b": "налог на добавленную стоимость",
    r"\bcit\b": "корпоративный подоходный налог",
}
_COMPILED = [(re.compile(p, re.I), v) for p, v in _TERMS.items()]


def expand(query: str) -> str:
    extra = [v for rx, v in _COMPILED if rx.search(query) and v.lower() not in query.lower()]
    return f"{query} ({'; '.join(dict.fromkeys(extra))})" if extra else query


_YEAR = re.compile(
    r"\b(?:в|с|на|до|по|за|in|since|from|for)?\s*(?:19|20)\d{2}\s*"
    r"(?:году|года|год|г\.|жылы|жылдан|жылғы|жылға|year)?(?=\W|$)",
    re.I,
)


def strip_years(query: str) -> str:
    """Remove "в 2025 году" / "2026 жылы" / "in 2026": years select the code version, they are not topic words."""
    stripped = re.sub(r"\s{2,}", " ", _YEAR.sub(" ", query)).strip()
    return stripped or query
