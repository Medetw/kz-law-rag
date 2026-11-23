"""Shared fixtures: adilet-like HTML, a fake retriever and a scripted chat model (no network, no models)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from kzlaw.config import Settings
from kzlaw.corpus import codes_in_force
from kzlaw.db import StoredChunk
from kzlaw.embeddings import HashEmbedder
from kzlaw.search.hybrid import Hit
from kzlaw.search.text import tokenize

# Structure copied from adilet.zan.kz (Tax Code 2026, RU): notes, red editorial remarks, h3 headings,
# article headers in <p><b>, a table inside an article.
TAX_RU_HTML = """<!DOCTYPE html><article><p>
<font color="#FF0000">&nbsp;&nbsp; <a name="z4"></a>Примечание ИЗПИ!</font><br></p>
<span class="note">&nbsp; Сноска. По всему тексту слово "тенге" заменено словом "теңге";</span><br>
<h3 id="z6"> ОСОБЕННАЯ ЧАСТЬ</h3><h3 id="z7"> РАЗДЕЛ 7. НАЛОГ НА ДОБАВЛЕННУЮ СТОИМОСТЬ</h3>
<h3 id="z8"> Глава 51. ПОРЯДОК ИСЧИСЛЕНИЯ И УПЛАТЫ НАЛОГА</h3>
<p><b><a name="z8609"></a>Статья 503. Ставки налога на добавленную стоимость</b></p><p id="z8610">
&nbsp;&nbsp;&nbsp; 1. Если иное не установлено настоящей статьей, ставка налога на добавленную стоимость
составляет 16 процентов и применяется к размеру облагаемого оборота и облагаемого импорта.</p>
<span class="note">Сноска. Статья 503 с изменением, внесенным Законом РК от 11.06.2026.</span>
<p id="z8611">&nbsp; 2. Ставка налога на добавленную стоимость составляет с 1 января 2026 года – 5 процентов.</p>
<p><b><a name="z8620"></a>Статья 504. Налоговый период</b></p><p id="z8621">
&nbsp;&nbsp;&nbsp; Налоговым периодом по налогу на добавленную стоимость является календарный квартал.</p>
<h3 id="z9"> Глава 52. ГРУППЫ</h3>
<p><b><a name="z4750"></a>Статья 276. Учет фиксированных активов</b></p><p id="z4751">
&nbsp; 1. Учет осуществляется по группам:</p>
<p><table width="615"><tr><td><p>№ группы</p></td><td><p>Наименование</p></td></tr>
<tr><td><p>I</p></td><td><p>Здания, сооружения</p></td></tr></table></p>
</article>"""

TAX_KZ_HTML = """<!DOCTYPE html><article>
<h3 id="z6"> ЕРЕКШЕ БӨЛІК</h3><h3 id="z7"> 7-БӨЛІМ. ҚОСЫЛҒАН ҚҰН САЛЫҒЫ</h3>
<h3 id="z8611"> 503-бап. Қосылған құн салығының мөлшерлемелері</h3><p id="z8612">
&nbsp; 1.&nbsp;Егер осы бапта өзгеше белгіленбесе, қосылған құн салығының мөлшерлемесі 16 пайызды құрайды.</p>
</article>"""


def chunk(code: str, article: str, title: str, text: str, lang: str = "ru", part: int = 0) -> StoredChunk:
    return StoredChunk(
        id=f"{code}:{article}:{lang}:{part}",
        code=code,
        article=article,
        lang=lang,  # type: ignore[arg-type]
        title=title,
        heading=f"{code}. Статья {article}. {title}",
        text=text,
        points=["1"],
        url=f"https://adilet.zan.kz/rus/docs/X#{article}",
    )


CHUNKS = [
    chunk(
        "tax2026",
        "503",
        "Ставки налога на добавленную стоимость",
        "1. Ставка налога на добавленную стоимость составляет 16 процентов.",
    ),
    chunk(
        "tax2026",
        "504",
        "Налоговый период",
        "Налоговым периодом по налогу на добавленную стоимость является календарный квартал.",
    ),
    chunk(
        "tax2017",
        "422",
        "Ставки налога на добавленную стоимость",
        "1. Ставка налога на добавленную стоимость составляет 12 процентов.",
    ),
    chunk(
        "labor",
        "88",
        "Продолжительность основного оплачиваемого ежегодного трудового отпуска",
        "Отпуск предоставляется продолжительностью двадцать четыре календарных дня.",
    ),
]


class FakeRetriever:
    """Keyword-overlap ranking over a few chunks, with the same interface the tools use."""

    def __init__(self, chunks: Sequence[StoredChunk] = CHUNKS) -> None:
        self.chunks = list(chunks)
        self.embedder = HashEmbedder()
        self.calls: list[dict[str, Any]] = []

    def search(
        self,
        query: str,
        k: int = 8,
        codes: list[str] | None = None,
        langs: list[str] | None = None,
        as_of: date | None = None,
        **_: Any,
    ) -> list[Hit]:
        self.calls.append({"query": query, "codes": codes, "as_of": as_of})
        allowed = codes or codes_in_force(as_of or date(2026, 6, 1))
        q = set(tokenize(query))
        scored = [(len(q & set(tokenize(c.full_text))), c) for c in self.chunks if c.code in allowed]
        scored = [(s, c) for s, c in scored if s > 0]
        scored.sort(key=lambda x: -x[0])
        return [Hit(c, float(s)) for s, c in scored[:k]]

    def article(self, code: str, article: str, lang: str = "ru") -> list[StoredChunk]:
        return [c for c in self.chunks if c.code == code and c.article == article and c.lang == lang]


class ScriptedChatModel(BaseChatModel):
    """Returns the given AIMessages in order; records the prompts it saw."""

    responses: list[AIMessage]
    seen: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:  # type: ignore[override]
        return self

    def _generate(
        self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])


def tool_call(name: str, **args: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{name}", "type": "tool_call"}])


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, embedding_backend="hash", deepseek_api_key=None, langfuse_public_key=None)  # type: ignore[call-arg]


@pytest.fixture
def fake_retriever() -> FakeRetriever:
    return FakeRetriever()
