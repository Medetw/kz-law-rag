from datetime import date

from langchain_core.messages import AIMessage, HumanMessage

from kzlaw.agent.graph import answer_question
from kzlaw.config import Settings

from .conftest import FakeRetriever, ScriptedChatModel, tool_call


def test_answer_with_valid_citation(settings: Settings, fake_retriever: FakeRetriever) -> None:
    llm = ScriptedChatModel(
        responses=[
            tool_call("search_law", query="ставка налога на добавленную стоимость"),
            AIMessage(content="Основная ставка НДС — 16% [tax2026:503]."),
        ]
    )
    res = answer_question(
        "Какая ставка НДС?", as_of=date(2026, 6, 1), settings=settings, retriever=fake_retriever, llm=llm
    )  # type: ignore[arg-type]
    assert [c.cite_id for c in res.citations] == ["tax2026:503"]
    assert res.citations[0].url.startswith("https://adilet.zan.kz/")
    assert res.invalid_citations == []
    assert "tax2026:503" in res.retrieved
    assert res.tool_calls[0].startswith("search_law(")


def test_invented_citation_gets_one_retry_then_is_removed(settings: Settings, fake_retriever: FakeRetriever) -> None:
    llm = ScriptedChatModel(
        responses=[
            tool_call("search_law", query="ставка налога на добавленную стоимость"),
            AIMessage(content="Ставка 16% [tax2026:503], см. также [tax2026:999]."),
            AIMessage(content="Ставка 16% [tax2026:503] и [tax2026:999]."),  # still cites an unseen article
        ]
    )
    res = answer_question("Какая ставка НДС?", settings=settings, retriever=fake_retriever, llm=llm)  # type: ignore[arg-type]
    assert res.invalid_citations == ["tax2026:999"]
    assert "999" not in res.answer
    assert [c.cite_id for c in res.citations] == ["tax2026:503"]
    retry_prompt = llm.seen[-1][-1]
    assert isinstance(retry_prompt, HumanMessage)
    assert "tax2026:999" in str(retry_prompt.content)


def test_as_of_date_selects_the_old_code(settings: Settings, fake_retriever: FakeRetriever) -> None:
    llm = ScriptedChatModel(
        responses=[
            tool_call("search_law", query="ставка налога на добавленную стоимость"),
            AIMessage(content="В 2025 году ставка была 12% [tax2017:422]."),
        ]
    )
    res = answer_question(
        "Какая ставка НДС была в 2025?", as_of=date(2025, 6, 1), settings=settings, retriever=fake_retriever, llm=llm
    )  # type: ignore[arg-type]
    assert fake_retriever.calls[0]["as_of"] == date(2025, 6, 1)
    assert [c.cite_id for c in res.citations] == ["tax2017:422"]


def test_get_article_and_compare_tools(settings: Settings, fake_retriever: FakeRetriever) -> None:
    llm = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "get_article", "args": {"code": "labor", "article": "88"}, "id": "a", "type": "tool_call"},
                    {
                        "name": "compare_editions",
                        "args": {"query": "ставка налога на добавленную стоимость"},
                        "id": "b",
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(content="24 дня [labor:88]; НДС был 12% [tax2017:422], стал 16% [tax2026:503]."),
        ]
    )
    res = answer_question("Отпуск и НДС?", settings=settings, retriever=fake_retriever, llm=llm)  # type: ignore[arg-type]
    assert {c.cite_id for c in res.citations} == {"labor:88", "tax2017:422", "tax2026:503"}
    assert res.invalid_citations == []


def test_bad_tool_arguments_do_not_crash_the_run(settings: Settings, fake_retriever: FakeRetriever) -> None:
    llm = ScriptedChatModel(
        responses=[
            tool_call("get_article", code="criminal", article="1"),
            AIMessage(content="В базе нет этого кодекса, ответ не найден."),
        ]
    )
    res = answer_question("Уголовный кодекс?", settings=settings, retriever=fake_retriever, llm=llm)  # type: ignore[arg-type]
    assert res.citations == []
    assert "не найден" in res.answer


def test_chunks_of_one_article_seen_in_two_calls_are_merged(settings: Settings) -> None:
    from .conftest import chunk

    retriever = FakeRetriever(
        [
            chunk("tax2026", "357", "Ставки налога", "1. Налог исчисляется по ставкам пункта 2.", part=0),
            chunk("tax2026", "357", "Ставки налога", "2. Ставка налога составляет 20 процентов.", part=1),
        ]
    )
    llm = ScriptedChatModel(
        responses=[
            tool_call("search_law", query="исчисляется по ставкам"),
            tool_call("search_law", query="составляет 20 процентов"),
            AIMessage(content="Ставка 20% [tax2026:357]."),
        ]
    )
    res = answer_question("Ставка КПН?", settings=settings, retriever=retriever, llm=llm)  # type: ignore[arg-type]
    assert "пункта 2" in res.contexts["tax2026:357"]
    assert "20 процентов" in res.contexts["tax2026:357"]


VAT_QUOTE = "«Ставка налога на добавленную стоимость составляет 16 процентов»"


def test_verbatim_quote_is_verified(settings: Settings, fake_retriever: FakeRetriever) -> None:
    llm = ScriptedChatModel(
        responses=[
            tool_call("search_law", query="ставка налога на добавленную стоимость"),
            AIMessage(content=f"По закону: {VAT_QUOTE} [tax2026:503]."),
        ]
    )
    res = answer_question("Ставка НДС?", settings=settings, retriever=fake_retriever, llm=llm)  # type: ignore[arg-type]
    assert res.quotes == ["Ставка налога на добавленную стоимость составляет 16 процентов"]
    assert res.invalid_quotes == []
    assert VAT_QUOTE in res.answer


def test_changed_quote_gets_one_retry(settings: Settings, fake_retriever: FakeRetriever) -> None:
    llm = ScriptedChatModel(
        responses=[
            tool_call("search_law", query="ставка налога на добавленную стоимость"),
            AIMessage(
                content="По закону: «ставка налога на добавленную стоимость равна шестнадцати процентам» [tax2026:503]."
            ),
            AIMessage(content=f"По закону: {VAT_QUOTE} [tax2026:503]."),
        ]
    )
    res = answer_question("Ставка НДС?", settings=settings, retriever=fake_retriever, llm=llm)  # type: ignore[arg-type]
    retry = llm.seen[-1][-1]
    assert isinstance(retry, HumanMessage)
    assert "равна шестнадцати процентам" in str(retry.content)
    assert "Ставка НДС?" in str(retry.content)  # the original question is repeated to keep its language
    assert "не упоминай" in str(retry.content)
    assert res.invalid_quotes == []
    assert len(res.quotes) == 1


def test_quote_still_wrong_after_retry_is_unquoted(settings: Settings, fake_retriever: FakeRetriever) -> None:
    wrong = "ставка налога на добавленную стоимость равна шестнадцати процентам"
    llm = ScriptedChatModel(
        responses=[
            tool_call("search_law", query="ставка налога на добавленную стоимость"),
            AIMessage(content=f"По закону: «{wrong}» [tax2026:503]."),
            AIMessage(content=f"По закону: «{wrong}» [tax2026:503]."),
        ]
    )
    res = answer_question("Ставка НДС?", settings=settings, retriever=fake_retriever, llm=llm)  # type: ignore[arg-type]
    assert res.invalid_quotes == [wrong]
    assert f"«{wrong}»" not in res.answer
    assert wrong in res.answer
    assert [c.cite_id for c in res.citations] == ["tax2026:503"]  # the citation itself is still valid


def test_refusals_are_not_quote_checked(settings: Settings, fake_retriever: FakeRetriever) -> None:
    llm = ScriptedChatModel(
        responses=[
            tool_call("search_law", query="гражданство"),
            AIMessage(
                content="В Налоговом и Трудовом кодексах ответа нет: это Закон «О гражданстве Республики Казахстан»."
            ),
        ]
    )
    res = answer_question("Как получить гражданство?", settings=settings, retriever=fake_retriever, llm=llm)  # type: ignore[arg-type]
    assert res.invalid_quotes == []
    assert "«О гражданстве Республики Казахстан»" in res.answer
