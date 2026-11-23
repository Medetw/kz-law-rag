"""MCP server: the same tools as the agent, for Claude Desktop / Claude Code / any MCP client.

The client's own model does the reasoning; this server only retrieves articles, so no LLM key is needed.

    kzlaw-mcp                 # stdio (Claude Desktop, Claude Code)
    kzlaw-mcp --http 8765     # streamable HTTP at http://127.0.0.1:8765/mcp
"""

from __future__ import annotations

import argparse
from functools import lru_cache

from mcp.server.mcpserver import MCPServer

from kzlaw import tools as T
from kzlaw.config import get_settings
from kzlaw.search.hybrid import Retriever, get_retriever

INSTRUCTIONS = (
    "Tools over the Tax Code of Kazakhstan (tax2026: in force from 2026; tax2017: in force 2018-2025) and the "
    "Labour Code (labor), in Russian and Kazakh, from adilet.zan.kz. Cite articles as [code:article], e.g. "
    "[tax2026:503], and give the article url. Answer only from returned text."
)

server = MCPServer(name="kz-law", instructions=INSTRUCTIONS)


@lru_cache(maxsize=1)
def _retriever() -> Retriever:
    return get_retriever(get_settings())


@server.tool()
def search_law(
    query: str, as_of: str | None = None, code: str | None = None, lang: str | None = None, k: int = 6
) -> list[T.Source]:
    """Hybrid (BM25 + vector) search over articles of the codes in force on `as_of` (YYYY-MM-DD, default today).
    code: tax2026 | tax2017 | labor (optional). lang: ru | kz (optional). Queries may be in RU, KZ or EN."""
    return T.search_law(_retriever(), query, as_of=as_of, code=code, lang=lang, k=k)


@server.tool()
def get_article(code: str, article: str, lang: str = "ru") -> T.Source | None:
    """Full text of one article, e.g. code="tax2026", article="503"."""
    return T.get_article(_retriever(), code, article, lang)


@server.tool()
def compare_editions(query: str, k: int = 3) -> T.EditionsComparison:
    """Top articles on a topic in the old Tax Code (2018-2025) and the new one (2026+), to explain what changed."""
    return T.compare_editions(_retriever(), query, k=k)


def main() -> None:
    parser = argparse.ArgumentParser(prog="kzlaw-mcp")
    parser.add_argument("--http", type=int, metavar="PORT", help="serve streamable HTTP instead of stdio")
    args = parser.parse_args()
    if args.http:
        server.run("streamable-http", port=args.http)
    else:
        server.run("stdio")


if __name__ == "__main__":
    main()
