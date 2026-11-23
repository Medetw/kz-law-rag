"""Golden set loading and retrieval metrics shared by the eval scripts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"
EVAL_DATE = date(2026, 6, 1)  # fixed "today" so results do not drift with the calendar


@dataclass
class Item:
    id: str
    lang: str
    type: str
    question: str
    expected: list[list[str]]
    facts: list[str]
    as_of: date = EVAL_DATE
    codes: list[str] | None = field(default=None)


def load_golden(path: Path = HERE / "golden.jsonl") -> list[Item]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        as_of = date.fromisoformat(raw.pop("as_of")) if "as_of" in raw else EVAL_DATE
        item = Item(**raw, as_of=as_of)
        if item.type == "change":
            item.codes = ["tax2026", "tax2017"]
        items.append(item)
    return items


def group_recall(ranked: list[str], expected: list[list[str]], k: int) -> float:
    """Share of expected groups with at least one acceptable article in the top k."""
    top = set(ranked[:k])
    return sum(1 for g in expected if top.intersection(g)) / len(expected)


def reciprocal_rank(ranked: list[str], expected: list[list[str]], k: int = 10) -> float:
    """1 / rank of the first relevant article (any group), 0 if none in the top k."""
    relevant = {a for g in expected for a in g}
    for i, art in enumerate(ranked[:k], start=1):
        if art in relevant:
            return 1.0 / i
    return 0.0
