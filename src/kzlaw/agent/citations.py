"""Find and validate ``[code:article]`` citations in an answer."""

from __future__ import annotations

import re

_BRACKET = re.compile(r"\[([^\[\]]{3,120})\]")
_ID = re.compile(r"\b(tax2026|tax2017|labor)\s*:\s*(\d+(?:-\d+)?)")


def extract(text: str) -> list[str]:
    """Citation ids in order of first appearance: ``[tax2026:503]`` or ``[tax2026:503, labor:88]``."""
    found: dict[str, None] = {}
    for bracket in _BRACKET.findall(text):
        for code, art in _ID.findall(bracket):
            found[f"{code}:{art}"] = None
    return list(found)


def strip(text: str, invalid: set[str]) -> str:
    """Remove invalid ids from brackets; drop brackets left empty."""
    if not invalid:
        return text

    def fix(m: re.Match[str]) -> str:
        inner = m.group(1)
        ids = [f"{c}:{a}" for c, a in _ID.findall(inner)]
        if not ids:
            return m.group(0)
        keep = [i for i in ids if i not in invalid]
        return f"[{', '.join(keep)}]" if keep else ""

    return re.sub(r"\s+([.,;])", r"\1", _BRACKET.sub(fix, text))
