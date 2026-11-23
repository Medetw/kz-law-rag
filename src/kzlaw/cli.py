"""Command line: ``kzlaw ingest | search | ask``."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

from kzlaw.config import get_settings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="kzlaw")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="download codes from adilet, chunk, embed, store in pgvector")
    p_ing.add_argument("--code", action="append", help="code id (tax2026, tax2017, labor); default: all")
    p_ing.add_argument("--reset", action="store_true", help="drop and rebuild the index")

    p_ex = sub.add_parser("export-chunks", help="write chunks to JSONL for embedding on a GPU box")
    p_ex.add_argument("out", type=Path)
    p_lv = sub.add_parser("load-vectors", help="store vectors made by scripts/embed_gpu.py")
    p_lv.add_argument("vectors", type=Path)
    p_lv.add_argument("--reset", action="store_true")

    p_s = sub.add_parser("search", help="hybrid search, prints top hits")
    p_s.add_argument("query")
    p_s.add_argument("-k", type=int, default=5)
    p_s.add_argument("--mode", choices=["hybrid", "vector", "bm25"], default="hybrid")
    p_s.add_argument("--rerank", action="store_true")
    p_s.add_argument("--as-of", type=date.fromisoformat, default=None)

    p_a = sub.add_parser("ask", help="ask the agent a question")
    p_a.add_argument("question")
    p_a.add_argument("--as-of", type=date.fromisoformat, default=None)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = get_settings()

    if args.cmd == "ingest":
        from kzlaw.ingest.pipeline import ingest

        n = ingest(settings, args.code, reset=args.reset)
        print(f"embedded {n} new chunks")
    elif args.cmd == "export-chunks":
        from kzlaw.ingest.pipeline import export_chunks

        print(f"exported {export_chunks(settings, args.out)} chunks to {args.out}")
    elif args.cmd == "load-vectors":
        from kzlaw.ingest.pipeline import load_vectors

        print(f"stored {load_vectors(settings, args.vectors, reset=args.reset)} chunks")
    elif args.cmd == "search":
        from kzlaw.search.hybrid import get_retriever

        hits = get_retriever(settings).search(
            args.query, k=args.k, mode=args.mode, rerank=args.rerank, as_of=args.as_of
        )
        for h in hits:
            print(f"{h.score:7.4f}  {h.chunk.cite_id:14} {h.chunk.lang}  {h.chunk.title[:80]}")
    elif args.cmd == "ask":
        from kzlaw.agent.graph import answer_question

        result = answer_question(args.question, as_of=args.as_of, settings=settings)
        print(result.answer)
        print(json.dumps([c.model_dump() for c in result.citations], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
