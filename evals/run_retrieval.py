"""Retrieval eval: recall@k and MRR at the article level, for each retrieval configuration.

uv run python evals/run_retrieval.py            # all configs
uv run python evals/run_retrieval.py --no-rerank
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import RESULTS, Item, group_recall, load_golden, reciprocal_rank
from kzlaw.config import get_settings
from kzlaw.search.hybrid import Retriever

CONFIGS = [
    ("BM25", dict(mode="bm25", expand=False, rerank=False)),
    ("Vector (e5-small)", dict(mode="vector", expand=False, rerank=False)),
    ("Hybrid (RRF)", dict(mode="hybrid", expand=False, rerank=False)),
    ("Hybrid + query expansion", dict(mode="hybrid", expand=True, rerank=False)),
    ("Hybrid + expansion + reranker", dict(mode="hybrid", expand=True, rerank=True)),
]


def ranked_articles(r: Retriever, item: Item, **kw: object) -> list[str]:
    hits = r.search(item.question, k=40, codes=item.codes, as_of=item.as_of, **kw)  # type: ignore[arg-type]
    return list(dict.fromkeys(h.chunk.cite_id for h in hits))


def evaluate(r: Retriever, items: list[Item], **kw: object) -> dict[str, object]:
    rows = []
    for it in items:
        t0 = time.perf_counter()
        ranked = ranked_articles(r, it, **kw)
        ms = (time.perf_counter() - t0) * 1000
        rows.append(
            {
                "id": it.id,
                "lang": it.lang,
                "r1": group_recall(ranked, it.expected, 1),
                "r5": group_recall(ranked, it.expected, 5),
                "r10": group_recall(ranked, it.expected, 10),
                "rr": reciprocal_rank(ranked, it.expected),
                "ms": ms,
                "top5": ranked[:5],
            }
        )

    def agg(sub: list[dict[str, object]], key: str) -> float:
        return round(statistics.mean(float(x[key]) for x in sub), 3) if sub else float("nan")  # type: ignore[arg-type]

    out: dict[str, object] = {k: agg(rows, k) for k in ("r1", "r5", "r10", "rr")}
    out["by_lang_r5"] = {lang: agg([x for x in rows if x["lang"] == lang], "r5") for lang in ("ru", "kz", "en")}
    out["p50_ms"] = round(statistics.median(float(x["ms"]) for x in rows))  # type: ignore[arg-type]
    out["rows"] = rows
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rerank", action="store_true")
    args = ap.parse_args()
    items = [i for i in load_golden() if i.expected]  # out-of-scope questions have no relevant article
    r = Retriever(get_settings())
    configs = [c for c in CONFIGS if not (args.no_rerank and c[1]["rerank"])]
    results = {}
    print(f"{len(items)} questions with a known answer ({len(r.chunks)} chunks indexed)\n")
    header = f"| {'Retrieval':32} | R@1 | R@5 | R@10 | MRR | R@5 ru | R@5 kz | R@5 en | p50 ms |"
    lines = [header, "|" + "|".join("-" * (len(c)) for c in header.split("|")[1:-1]) + "|"]
    for name, kw in configs:
        res = evaluate(r, items, **kw)
        results[name] = res
        bl = res["by_lang_r5"]
        assert isinstance(bl, dict)
        lines.append(
            f"| {name:32} | {res['r1']:.2f} | {res['r5']:.2f} | {res['r10']:.2f} | {res['rr']:.2f} | "
            f"{bl['ru']:.2f} | {bl['kz']:.2f} | {bl['en']:.2f} | {res['p50_ms']} |"
        )
        print(lines[-1] if len(lines) > 3 else "\n".join(lines), flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "retrieval.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    (RESULTS / "retrieval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nsaved {RESULTS / 'retrieval.md'}")


if __name__ == "__main__":
    main()
