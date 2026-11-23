from typing import Any

import pytest
from fastapi.testclient import TestClient

from kzlaw import mcp_server
from kzlaw.api.main import create_app
from kzlaw.config import Settings

from .conftest import FakeRetriever


@pytest.fixture
def client(settings: Settings, fake_retriever: FakeRetriever) -> Any:
    with TestClient(create_app(settings, retriever=fake_retriever)) as c:  # type: ignore[arg-type]
        yield c


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["chunks"] == 4
    assert set(body["codes"]) == {"tax2026", "tax2017", "labor"}


def test_search_returns_distinct_articles_with_links(client: TestClient) -> None:
    hits = client.get("/search", params={"q": "ставка НДС", "k": 3}).json()
    assert hits[0]["cite_id"] == "tax2026:503"
    assert hits[0]["label"] == "НК РК 2026, ст. 503"
    assert len({h["cite_id"] for h in hits}) == len(hits)


def test_search_respects_as_of(client: TestClient) -> None:
    hits = client.get("/search", params={"q": "ставка НДС", "as_of": "2025-03-01"}).json()
    assert [h["cite_id"] for h in hits] == ["tax2017:422"]


def test_unknown_code_is_a_client_error(client: TestClient) -> None:
    assert client.get("/search", params={"q": "штраф", "code": "criminal"}).status_code == 422
    assert client.get("/articles/criminal/1").status_code == 404
    assert client.get("/articles/labor/9999").status_code == 404


def test_article_lookup(client: TestClient) -> None:
    body = client.get("/articles/labor/88").json()
    assert body["cite_id"] == "labor:88"
    assert "двадцать четыре" in body["text"]


def test_ask_without_llm_key_is_503_not_500(client: TestClient) -> None:
    r = client.post("/ask", json={"question": "Какая ставка НДС?"})
    assert r.status_code == 503
    assert "DEEPSEEK_API_KEY" in r.json()["detail"]


def test_demo_page_search_mode_works_without_llm(client: TestClient) -> None:
    html = client.get("/", params={"q": "ставка НДС", "mode": "search"}).text
    assert "НК РК 2026, ст. 503" in html
    assert "отвечает ИИ" in html


async def test_mcp_tools_listed_and_callable(monkeypatch: pytest.MonkeyPatch, fake_retriever: FakeRetriever) -> None:
    monkeypatch.setattr(mcp_server, "_retriever", lambda: fake_retriever)
    names = {t.name for t in await mcp_server.server.list_tools()}
    assert names == {"search_law", "get_article", "compare_editions"}
    result = await mcp_server.server.call_tool("search_law", {"query": "ставка НДС", "k": 2})
    assert not getattr(result, "isError", getattr(result, "is_error", False))
    assert "tax2026:503" in str(result)
    cmp = await mcp_server.server.call_tool("compare_editions", {"query": "ставка НДС"})
    assert "tax2017:422" in str(cmp)


def test_llm_outage_is_502_and_a_readable_page(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx
    import openai

    from kzlaw.api import main

    def boom(*args: object, **kwargs: object) -> None:
        raise openai.APIConnectionError(request=httpx.Request("POST", "https://api.deepseek.com/chat/completions"))

    monkeypatch.setattr(main, "answer_question", boom)
    r = client.post("/ask", json={"question": "Какая ставка НДС?"})
    assert r.status_code == 502
    assert "APIConnectionError" in r.json()["detail"]
    page = client.get("/", params={"q": "Какая ставка НДС?"})
    assert page.status_code == 200
    assert "недоступна" in page.text
