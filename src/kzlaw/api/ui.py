"""Server-rendered demo page (no JS framework): one form, answer with clickable citations."""

from __future__ import annotations

import html
import re

from kzlaw.agent.graph import AnswerResult
from kzlaw.tools import Source

EXAMPLES = [
    "Какая ставка НДС в 2026 году?",
    "Как изменился порог регистрации по НДС в новом кодексе?",
    "Жыл сайынғы ақылы еңбек демалысы неше күн?",
    "How long can a probation period be?",
]

_CSS = """
:root{--bg:#f7f7f5;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#e3e2dc;--accent:#0b6e4f;--warn:#8a5a00}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--muted:#a3a29b;--card:#1f1f1d;--line:#34332f;
--accent:#5cc49b;--warn:#e0b04f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:860px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:22px;margin:0 0 4px}.sub{color:var(--muted);margin:0 0 20px;font-size:14px}
form{display:flex;gap:8px;flex-wrap:wrap}input[type=text]{flex:1 1 420px;padding:10px 12px;font-size:16px;
border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg)}
button{padding:10px 16px;border:0;border-radius:8px;background:var(--accent);color:#fff;font-size:15px;cursor:pointer}
label{color:var(--muted);font-size:14px;display:flex;gap:6px;align-items:center}
.ex{margin:10px 0 24px;font-size:14px;color:var(--muted)}.ex a{color:var(--accent);margin-right:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:12px 0}
.answer{white-space:pre-wrap}.cite{color:var(--accent);font-weight:600;text-decoration:none}
.meta{color:var(--muted);font-size:13px}.src h3{margin:0 0 4px;font-size:15px}.src p{margin:6px 0 0;font-size:14px}
.warn{color:var(--warn);font-size:13px;margin-top:12px}.err{border-color:var(--warn)}
code{font-size:13px}
"""


def _link_citations(answer: str, result: AnswerResult) -> str:
    urls = {c.cite_id: (c.url, c.label) for c in result.citations}

    def repl(m: re.Match[str]) -> str:
        cid = m.group(0)
        if cid in urls:
            url, label = urls[cid]
            return f'<a class="cite" href="{html.escape(url)}" title="{html.escape(label)}">{cid}</a>'
        return cid

    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(answer))  # the models' markdown bold
    return re.sub(r"(?:tax2026|tax2017|labor):\d+(?:-\d+)?", repl, text)


def _source_card(s: Source) -> str:
    snippet = html.escape(s.text[:420] + ("…" if len(s.text) > 420 else ""))
    return (
        f'<div class="card src"><h3><a class="cite" href="{html.escape(s.url)}">{html.escape(s.label)}</a> '
        f"· {html.escape(s.title)}</h3><div class=meta>{s.cite_id} · {s.lang}"
        f"{' · пп. ' + ', '.join(s.points) if s.points else ''}</div><p>{snippet}</p></div>"
    )


def page(
    q: str,
    mode: str,
    result: AnswerResult | None = None,
    sources: list[Source] | None = None,
    error: str | None = None,
    disclaimer: str | None = None,
) -> str:
    checked = " checked" if mode == "search" else ""
    examples = " ".join(f'<a href="/?q={html.escape(e)}">{html.escape(e)}</a>' for e in EXAMPLES)
    body = ""
    if error:
        body += f'<div class="card err">{html.escape(error)}</div>'
    if result is not None:
        body += f'<div class="card"><div class="answer">{_link_citations(result.answer, result)}</div>'
        if result.citations:
            refs = "".join(
                f'<li><a class="cite" href="{html.escape(c.url)}">{html.escape(c.label)}</a> · '
                f"{html.escape(c.title)}</li>"
                for c in result.citations
            )
            body += f"<ul class=meta>{refs}</ul>"
        body += (
            f'<div class="meta">на дату {result.as_of} · инструменты: {html.escape("; ".join(result.tool_calls))}'
            f"{' · удалены ссылки: ' + ', '.join(result.invalid_citations) if result.invalid_citations else ''}"
            f"{' · цитат сверено с текстом: ' + str(len(result.quotes)) if result.quotes else ''}"
            f"{' · не дословно, кавычки сняты: ' + str(len(result.invalid_quotes)) if result.invalid_quotes else ''}"
            "</div>"
        )
        if disclaimer:
            body += f'<div class="warn">{html.escape(disclaimer)}</div>'
        body += "</div>"
    for s in sources or []:
        body += _source_card(s)
    return f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>kz-law-rag</title>
<style>{_CSS}</style></head>
<body><main><h1>kz-law-rag</h1>
<p class="sub">Налоговый и Трудовой кодексы РК (RU/KZ) · ответы со ссылками на статьи adilet.zan.kz · отвечает ИИ</p>
<form method="get" action="/"><input type="text" name="q" value="{html.escape(q)}"
placeholder="Вопрос на русском, казахском или английском" autofocus>
<label><input type="checkbox" name="mode" value="search"{checked}>Только поиск</label><button>Спросить</button></form>
<div class="ex">Примеры: {examples}</div>{body}</main></body></html>"""
