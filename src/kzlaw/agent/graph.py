"""LangGraph agent.

    START -> agent --(tool calls)--> tools -> agent ...
                   \\--(final text)--> check_citations --(invalid ids, 1st time)--> agent
                                                     \\--(ok)--> check_quotes --(not verbatim, 1st time)--> agent
                                                                                \\--(ok)--> END

* ``agent``: the chat model with three tools bound (search_law, get_article, compare_editions).
* ``tools``: runs the calls; every returned article is stored in ``state.sources`` by its citation id.
* ``check_citations``: every ``[code:article]`` in the answer must be in ``state.sources``. If not, the model
  gets one chance to fix it; after that invalid citations are removed, so the API never shows an article
  the model did not actually read.
* ``check_quotes``: every «quote» of 4+ words must appear word for word in a retrieved article (conditions and
  deadlines are quoted, see prompts.py). One retry; then the quote marks are dropped and the quote is reported.
"""

from __future__ import annotations

import operator
from datetime import date
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from kzlaw import tools as T
from kzlaw.agent import citations, quotes
from kzlaw.agent.llm import get_chat_model
from kzlaw.agent.prompts import CITATION_RETRY, QUOTE_RETRY, system_prompt
from kzlaw.config import Settings, get_settings
from kzlaw.corpus import CODES
from kzlaw.search.hybrid import Retriever, get_retriever
from kzlaw.tracing import flush, trace

MAX_TOOL_ROUNDS = 4
MAX_CITATION_RETRIES = 1
MAX_QUOTE_RETRIES = 1


def _merge_source(old: T.Source, new: T.Source) -> T.Source:
    """Two pieces of the same article (different chunks / full text): keep everything the model saw."""
    if new.text in old.text:
        return old
    if old.text in new.text:
        return new
    points = list(dict.fromkeys([*old.points, *new.points]))
    return old.model_copy(update={"text": f"{old.text}\n…\n{new.text}", "points": points})


def _merge(a: dict[str, T.Source], b: dict[str, T.Source]) -> dict[str, T.Source]:
    out = dict(a)
    for cid, src in b.items():
        out[cid] = _merge_source(out[cid], src) if cid in out else src
    return out


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    as_of: str
    sources: Annotated[dict[str, T.Source], _merge]
    tool_rounds: Annotated[int, operator.add]
    retries: Annotated[int, operator.add]
    quote_retries: Annotated[int, operator.add]
    answer: str
    cited: list[str]
    invalid: list[str]
    quotes_ok: list[str]
    quotes_bad: list[str]


class SearchArgs(BaseModel):
    query: str = Field(description="что искать, лучше формулировками закона: 'ставка НДС', 'ежегодный отпуск'")
    as_of: str | None = Field(None, description="дата YYYY-MM-DD, на которую нужен закон; по умолчанию дата вопроса")
    code: str | None = Field(None, description="tax2026 | tax2017 | labor; по умолчанию все действующие на дату")
    lang: str | None = Field(None, description="ru | kz; по умолчанию оба языка")


class ArticleArgs(BaseModel):
    code: str = Field(description="tax2026 | tax2017 | labor")
    article: str = Field(description="номер статьи, например '503' или '88'")
    lang: str = Field("ru", description="ru | kz")


class CompareArgs(BaseModel):
    query: str = Field(description="тема, например 'ставка НДС' или 'порог регистрации по НДС'")


def make_tools(retriever: Retriever, default_as_of: str) -> list[BaseTool]:
    def search(
        query: str, as_of: str | None = None, code: str | None = None, lang: str | None = None
    ) -> tuple[str, list[T.Source]]:
        found = T.search_law(retriever, query, as_of=as_of or default_as_of, code=code, lang=lang)
        return T.format_sources(found), found

    def article(code: str, article: str, lang: str = "ru") -> tuple[str, list[T.Source]]:
        if code not in CODES:
            return f"Неизвестный код {code!r}. Есть: {', '.join(CODES)}", []
        src = T.get_article(retriever, code, article, lang)
        return (T.format_sources([src]), [src]) if src else (f"Статья {code}:{article} не найдена.", [])

    def compare(query: str) -> tuple[str, list[T.Source]]:
        cmp = T.compare_editions(retriever, query)
        text = (
            "НК РК 2017 (действовал 2018-2025):\n"
            + T.format_sources(cmp.tax2017)
            + "\n\n=====\n\nНК РК 2026 (действует с 2026):\n"
            + T.format_sources(cmp.tax2026)
        )
        return text, [*cmp.tax2017, *cmp.tax2026]

    return [
        StructuredTool.from_function(
            search,
            name="search_law",
            args_schema=SearchArgs,
            response_format="content_and_artifact",
            description="Гибридный поиск (BM25 + векторы) по статьям кодексов, действующих на дату.",
        ),
        StructuredTool.from_function(
            article,
            name="get_article",
            args_schema=ArticleArgs,
            response_format="content_and_artifact",
            description="Полный текст одной статьи по коду и номеру.",
        ),
        StructuredTool.from_function(
            compare,
            name="compare_editions",
            args_schema=CompareArgs,
            response_format="content_and_artifact",
            description="Как тема регулировалась в НК 2017 (до 2026) и как в НК 2026: для вопросов 'что изменилось'.",
        ),
    ]


def _question(state: AgentState) -> str:
    """The user's original question, repeated in retry notes so the rewrite keeps its language."""
    return str(state["messages"][0].content)


def build_graph(llm: BaseChatModel, retriever: Retriever, as_of: str) -> Any:
    tools = make_tools(retriever, as_of)
    by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    def agent(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        system = SystemMessage(system_prompt(date.today().isoformat(), state["as_of"]))  # noqa: DTZ011
        model = llm if state["tool_rounds"] >= MAX_TOOL_ROUNDS else llm_with_tools
        return {"messages": [model.invoke([system, *state["messages"]], config)]}

    def run_tools(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage):
            return {}
        out: list[ToolMessage] = []
        found: dict[str, T.Source] = {}
        for call in last.tool_calls:
            tool = by_name.get(call["name"])
            if tool is None:
                out.append(ToolMessage(f"Нет инструмента {call['name']}", tool_call_id=call["id"]))
                continue
            try:
                msg = tool.invoke({**call, "type": "tool_call"}, config)
            except Exception as e:  # bad arguments from the model: report back instead of crashing the run
                out.append(ToolMessage(f"Ошибка инструмента: {e}", tool_call_id=call["id"], status="error"))
                continue
            out.append(msg)
            for src in msg.artifact or []:
                found[src.cite_id] = _merge_source(found[src.cite_id], src) if src.cite_id in found else src
        return {"messages": out, "sources": found, "tool_rounds": 1}

    def check_citations(state: AgentState) -> dict[str, Any]:
        text = str(state["messages"][-1].content)
        cited = citations.extract(text)
        bad = [c for c in cited if c not in state["sources"]]
        if bad and state["retries"] < MAX_CITATION_RETRIES:
            note = CITATION_RETRY.format(bad=", ".join(bad), question=_question(state))
            return {"messages": [HumanMessage(note)], "retries": 1}
        return {
            "answer": citations.strip(text, set(bad)),
            "cited": [c for c in cited if c not in bad],
            "invalid": bad,
        }

    def check_quotes(state: AgentState) -> dict[str, Any]:
        text = state["answer"]
        if not state["cited"]:  # refusals and answers without sources: nothing to check quotes against
            return {"quotes_ok": [], "quotes_bad": []}
        res = quotes.check(text, [f"{s.title}\n{s.text}" for s in state["sources"].values()])
        if res.invalid and state["quote_retries"] < MAX_QUOTE_RETRIES:
            bad = "; ".join(f"«{q}»" for q in res.invalid)
            note = QUOTE_RETRY.format(bad=bad, question=_question(state))
            return {"messages": [HumanMessage(note)], "quote_retries": 1, "answer": ""}
        return {"answer": quotes.unquote(text, res.invalid), "quotes_ok": res.verified, "quotes_bad": res.invalid}

    def after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else "check_citations"

    g = StateGraph(AgentState)
    g.add_node("agent", agent)
    g.add_node("tools", run_tools)
    g.add_node("check_citations", check_citations)
    g.add_node("check_quotes", check_quotes)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", after_agent, ["tools", "check_citations"])
    g.add_edge("tools", "agent")
    g.add_conditional_edges(
        "check_citations", lambda s: "check_quotes" if s["answer"] else "agent", ["agent", "check_quotes"]
    )
    g.add_conditional_edges("check_quotes", lambda s: END if s["answer"] else "agent", ["agent", END])
    return g.compile()


class Citation(BaseModel):
    cite_id: str
    label: str
    title: str
    url: str


class AnswerResult(BaseModel):
    question: str
    as_of: str
    answer: str
    citations: list[Citation]
    invalid_citations: list[str]
    quotes: list[str] = Field(default_factory=list)  # «quotes» verified word for word against the articles
    invalid_quotes: list[str] = Field(default_factory=list)  # not verbatim after one retry: shown unquoted
    retrieved: list[str]  # every article the tools returned, in order
    contexts: dict[str, str]  # cite_id -> text the model saw (used by the faithfulness eval)
    tool_calls: list[str]
    tokens: dict[str, int] = Field(default_factory=dict)  # summed over the agent's LLM calls
    trace_id: str | None = None


def answer_question(
    question: str,
    as_of: date | None = None,
    settings: Settings | None = None,
    retriever: Retriever | None = None,
    llm: BaseChatModel | None = None,
) -> AnswerResult:
    settings = settings or get_settings()
    day = (as_of or date.today()).isoformat()  # noqa: DTZ011
    graph = build_graph(llm or get_chat_model(settings), retriever or get_retriever(settings), day)
    meta = {"as_of": day, "llm": f"{settings.llm_provider}/{settings.resolved_llm_model}"}
    with trace(settings, "ask", input={"question": question, "as_of": day}, metadata=meta) as ctx:
        state = graph.invoke(
            {
                "messages": [HumanMessage(question)],
                "as_of": day,
                "sources": {},
                "tool_rounds": 0,
                "retries": 0,
                "quote_retries": 0,
                "answer": "",
                "cited": [],
                "invalid": [],
                "quotes_ok": [],
                "quotes_bad": [],
            },
            config={"callbacks": ctx.callbacks, "recursion_limit": 25, "run_name": "kz-law-agent"},
        )
        sources: dict[str, T.Source] = state["sources"]
        usage = [m.usage_metadata for m in state["messages"] if isinstance(m, AIMessage) and m.usage_metadata]
        result = AnswerResult(
            question=question,
            as_of=day,
            answer=state["answer"],
            citations=[
                Citation(cite_id=c, label=sources[c].label, title=sources[c].title, url=sources[c].url)
                for c in state["cited"]
            ],
            invalid_citations=state["invalid"],
            quotes=state["quotes_ok"],
            invalid_quotes=state["quotes_bad"],
            retrieved=list(sources),
            contexts={c: s.text for c, s in sources.items()},
            tool_calls=[
                f"{tc['name']}({', '.join(f'{k}={v!r}' for k, v in tc['args'].items())})"
                for m in state["messages"]
                if isinstance(m, AIMessage)
                for tc in m.tool_calls
            ],
            tokens={
                "input": sum(u["input_tokens"] for u in usage),
                "output": sum(u["output_tokens"] for u in usage),
            },
            trace_id=ctx.trace_id,
        )
        ctx.output = {"answer": result.answer, "citations": state["cited"], "invalid": state["invalid"]}
        ctx.metadata = {
            "retrieved": result.retrieved,
            "invalid_citations": state["invalid"],
            "quotes": len(state["quotes_ok"]),
            "invalid_quotes": state["quotes_bad"],
        }
    flush(settings)
    return result
