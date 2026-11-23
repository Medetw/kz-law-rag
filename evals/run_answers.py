"""Answer eval: run the LangGraph agent on the golden set and score the answers.

    uv run python evals/run_answers.py              # needs DEEPSEEK_API_KEY (or ANTHROPIC_API_KEY) in .env
    uv run python evals/run_answers.py --limit 5    # smoke run

Metrics (answerable questions unless noted):
* citation recall   - share of expected articles (groups) the answer cites
* citation hit      - answer cites at least one expected article
* fact accuracy     - every key fact regex (e.g. "16", "24") is in the answer
* faithfulness      - LLM judge: every claim is supported by the text of the cited articles
* refusal accuracy  - out-of-scope questions: "not found" and no citations
* caught citations  - answers where check_citations removed an article the model never retrieved
* quote accuracy    - share of «quotes» in final answers that match the article word for word
                      (quotes still wrong after the retry are shown unquoted and counted as misses)
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from langchain_core.messages import HumanMessage, SystemMessage

from common import RESULTS, Item, group_recall, load_golden
from kzlaw.agent.graph import AnswerResult, answer_question
from kzlaw.agent.llm import get_chat_model
from kzlaw.agent.prompts import NO_ANSWER_MARKERS
from kzlaw.config import Settings, get_settings
from kzlaw.corpus import CODES
from kzlaw.search.hybrid import Retriever

# USD per 1M tokens (input, output), list prices used in the author's other projects; cache hits ignored
PRICES_PER_M = {"deepseek-flash": (0.30, 1.20), "deepseek-v4-pro": (1.32, 3.96)}

JUDGE = """Ты проверяешь ответ ассистента по законам Казахстана.
Даны тексты статей, на которые ссылается ответ, и сам ответ.
Ответ "faithful", если каждое фактическое утверждение в нём (числа, сроки, условия) прямо следует из этих текстов.
Общие вводные фразы и оговорки не считаются утверждениями.
Верни только JSON: {"faithful": true|false, "unsupported": ["утверждение без опоры в тексте", ...]}"""


def _code_header(cid: str) -> str:
    code, art = cid.split(":")
    c = CODES[code]
    until = f"по {c.valid_to:%d.%m.%Y}" if c.valid_to else "по настоящее время"
    return f"{c.title_ru}, ст. {art} (кодекс действует с {c.valid_from:%d.%m.%Y} {until})"


def judge(settings: Settings, res: AnswerResult) -> dict[str, Any]:
    cited = {c.cite_id for c in res.citations}
    context = "\n\n".join(f"[{cid}] {_code_header(cid)}\n{text}" for cid, text in res.contexts.items() if cid in cited)
    if not cited:
        return {"faithful": None, "unsupported": []}
    msg = get_chat_model(settings).invoke(
        [SystemMessage(JUDGE), HumanMessage(f"СТАТЬИ:\n{context}\n\nВОПРОС: {res.question}\n\nОТВЕТ:\n{res.answer}")]
    )
    m = re.search(r"\{.*\}", str(msg.content), re.S)
    try:
        return dict(json.loads(m.group(0))) if m else {"faithful": None, "unsupported": ["judge: no JSON"]}
    except json.JSONDecodeError:
        return {"faithful": None, "unsupported": ["judge: bad JSON"]}


def run_one(settings: Settings, retriever: Retriever, it: Item) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        res = answer_question(it.question, as_of=it.as_of, settings=settings, retriever=retriever)
    except Exception as e:  # keep going: one failed call must not kill a 48-question run
        return {"id": it.id, "type": it.type, "lang": it.lang, "error": repr(e)}
    secs = time.perf_counter() - t0
    cited = [c.cite_id for c in res.citations]
    row: dict[str, Any] = {
        "id": it.id,
        "type": it.type,
        "lang": it.lang,
        "question": it.question,
        "answer": res.answer,
        "cited": cited,
        "invalid_removed": res.invalid_citations,
        "quotes": res.quotes,
        "invalid_quotes": res.invalid_quotes,
        "tool_calls": res.tool_calls,
        "seconds": round(secs, 1),
        "tokens": res.tokens,
        "trace_id": res.trace_id,
    }
    if it.expected:
        row["citation_recall"] = group_recall(cited, it.expected, k=len(cited) or 1)
        row["citation_hit"] = float(any(set(cited) & set(g) for g in it.expected))
        row["facts_ok"] = float(all(re.search(f, res.answer, re.I) for f in it.facts)) if it.facts else None
        j = judge(settings, res)
        row["faithful"] = j.get("faithful")
        row["unsupported"] = j.get("unsupported", [])
    else:
        said_no = any(m in res.answer.lower() for m in NO_ANSWER_MARKERS)
        row["refused_ok"] = float(said_no and not cited)
    return row


def _quote_accuracy(rows: list[dict[str, Any]]) -> float | None:
    good = sum(len(r["quotes"]) for r in rows)
    bad = sum(len(r["invalid_quotes"]) for r in rows)
    return round(good / (good + bad), 3) if good + bad else None


def mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return round(statistics.mean(vals), 3) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    settings = get_settings()
    get_chat_model(settings)  # fail fast without a key
    retriever = Retriever(settings)
    items = load_golden()[: args.limit] if args.limit else load_golden()
    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(lambda it: run_one(settings, retriever, it), items))
    ok = [r for r in rows if "error" not in r]
    answerable = [r for r in ok if r["type"] != "out_of_scope"]
    summary = {
        "llm": f"{settings.llm_provider}/{settings.resolved_llm_model}",
        "questions": len(items),
        "errors": len(rows) - len(ok),
        "citation_recall": mean(answerable, "citation_recall"),
        "citation_hit": mean(answerable, "citation_hit"),
        "fact_accuracy": mean(answerable, "facts_ok"),
        "faithfulness": mean(
            [{**r, "faithful": float(r["faithful"])} for r in answerable if r.get("faithful") is not None], "faithful"
        ),
        "refusal_accuracy": mean([r for r in ok if r["type"] == "out_of_scope"], "refused_ok"),
        "answers_with_removed_citations": sum(1 for r in ok if r["invalid_removed"]),
        "answers_with_quotes": sum(1 for r in answerable if r["quotes"] or r["invalid_quotes"]),
        "quote_accuracy": _quote_accuracy(answerable),
        "by_lang_citation_hit": {
            lang: mean([r for r in answerable if r["lang"] == lang], "citation_hit") for lang in ("ru", "kz", "en")
        },
        "by_type_citation_hit": {
            t: mean([r for r in answerable if r["type"] == t], "citation_hit") for t in ("fact", "date", "change")
        },
        "p50_seconds": statistics.median(r["seconds"] for r in ok) if ok else None,
        "avg_tool_calls": round(statistics.mean(len(r["tool_calls"]) for r in ok), 2) if ok else None,
        "avg_tokens_in": round(statistics.mean(r["tokens"].get("input", 0) for r in ok)) if ok else None,
        "avg_tokens_out": round(statistics.mean(r["tokens"].get("output", 0) for r in ok)) if ok else None,
    }
    price = PRICES_PER_M.get(settings.resolved_llm_model)
    if price and ok:
        cost = sum(r["tokens"].get("input", 0) * price[0] + r["tokens"].get("output", 0) * price[1] for r in ok) / 1e6
        summary["agent_cost_usd_per_question"] = round(cost / len(ok), 5)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "answers.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    for r in rows:
        if "error" in r:
            print("ERROR", r["id"], r["error"])
        elif (
            r.get("citation_hit") == 0
            or r.get("facts_ok") == 0
            or r.get("faithful") is False
            or r.get("refused_ok") == 0
        ):
            print(f"MISS {r['id']} cited={r['cited']} facts_ok={r.get('facts_ok')} faithful={r.get('faithful')}")


if __name__ == "__main__":
    main()
