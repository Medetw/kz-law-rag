"""Optional second stage: a cross-encoder reads (question, chunk) together and scores relevance.

Slower than the first stage (one model call per candidate), so it only sees the top 30 fused hits.
"""

from __future__ import annotations

from typing import Any


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self._model: Any = CrossEncoder(model_name, device="cpu", max_length=512)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self._model.predict([(query, t) for t in texts], batch_size=16)]
