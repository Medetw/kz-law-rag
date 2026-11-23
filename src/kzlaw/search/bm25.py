"""Okapi BM25 over an in-memory inverted index.

~7k chunks fit in a few MB, so the index is built from PostgreSQL at startup in about a second.
For millions of chunks this would move into the database (ParadeDB pg_search) or OpenSearch.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from kzlaw.search.text import tokenize


class BM25:
    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.n = len(docs)
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths = np.zeros(self.n, dtype=np.float32)
        for i, doc in enumerate(docs):
            toks = tokenize(doc)
            lengths[i] = len(toks)
            for tok, tf in Counter(toks).items():
                postings[tok].append((i, tf))
        avg = float(lengths.mean()) if self.n else 1.0  # empty index: scores() returns an empty array
        self._norm = k1 * (1 - b + b * lengths / max(avg, 1e-9))
        self._index: dict[str, tuple[npt.NDArray[np.int32], npt.NDArray[np.float32], float]] = {}
        for tok, plist in postings.items():
            ids = np.fromiter((d for d, _ in plist), dtype=np.int32, count=len(plist))
            tfs = np.fromiter((t for _, t in plist), dtype=np.float32, count=len(plist))
            idf = math.log(1 + (self.n - len(plist) + 0.5) / (len(plist) + 0.5))
            self._index[tok] = (ids, tfs, idf)

    def scores(self, query: str) -> npt.NDArray[np.float32]:
        out = np.zeros(self.n, dtype=np.float32)
        for tok in set(tokenize(query)):
            hit = self._index.get(tok)
            if hit is None:
                continue
            ids, tfs, idf = hit
            out[ids] += idf * tfs * (self.k1 + 1) / (tfs + self._norm[ids])
        return out

    def top(self, query: str, k: int, mask: npt.NDArray[np.bool_] | None = None) -> list[tuple[int, float]]:
        s = self.scores(query)
        if mask is not None:
            s = np.where(mask, s, 0.0)
        k = min(k, int((s > 0).sum()))
        if k == 0:
            return []
        idx = np.argpartition(-s, k - 1)[:k]
        idx = idx[np.argsort(-s[idx])]
        return [(int(i), float(s[i])) for i in idx]
