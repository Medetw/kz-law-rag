"""Text -> vector. Real model: multilingual E5 (RU/KZ/EN in one space). Tests: a hashing embedder."""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from kzlaw.config import Settings
from kzlaw.search.text import tokenize

Vector = npt.NDArray[np.float32]


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> Vector: ...

    def embed_query(self, text: str) -> Vector: ...


class E5Embedder:
    """intfloat/multilingual-e5-*: needs the "passage: " / "query: " prefixes it was trained with."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model: Any = SentenceTransformer(model_name, device="cpu")
        get_dim: Any = (
            getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        )
        self.dim = int(get_dim())

    def embed_documents(self, texts: list[str]) -> Vector:
        out = self._model.encode(
            [f"passage: {t}" for t in texts], batch_size=16, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(out, dtype=np.float32)

    def embed_query(self, text: str) -> Vector:
        out = self._model.encode([f"query: {text}"], normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(out[0], dtype=np.float32)


class OnnxE5Embedder:
    """The same E5 weights exported to ONNX, run on onnxruntime: no PyTorch in the API image.

    With the fp32 export (default) vectors match sentence-transformers up to float noise, so a corpus
    indexed on a GPU with sentence-transformers can be queried here. E5 = mean of the token vectors
    (ignoring padding), then L2-normalised.
    """

    def __init__(self, repo: str, file: str = "onnx/model.onnx", max_len: int = 512) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        variant = file.rsplit("/", 1)[-1].removesuffix(".onnx")
        self.name = repo if variant == "model" else f"{repo}@{variant}"
        self._tok = Tokenizer.from_file(hf_hub_download(repo, "onnx/tokenizer.json"))
        self._tok.enable_truncation(max_len)
        pad = self._tok.token_to_id("<pad>")
        self._tok.enable_padding(pad_id=pad if pad is not None else 1, pad_token="<pad>")  # noqa: S106 - a tokenizer token, not a password
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = os.cpu_count() or 2
        self._sess = ort.InferenceSession(hf_hub_download(repo, file), opts, providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self._sess.get_inputs()}
        self.dim = int(self._sess.get_outputs()[0].shape[-1])

    def _encode(self, texts: list[str], batch: int = 16) -> Vector:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # similar lengths -> less padding
        for start in range(0, len(order), batch):
            idx = order[start : start + batch]
            enc = self._tok.encode_batch([texts[i] for i in idx])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self._sess.run(None, feed)[0]
            m = mask[..., None].astype(np.float32)
            vec = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
            out[idx] = vec / np.linalg.norm(vec, axis=1, keepdims=True)
        return out

    def embed_documents(self, texts: list[str]) -> Vector:
        return self._encode([f"passage: {t}" for t in texts])

    def embed_query(self, text: str) -> Vector:
        return np.asarray(self._encode([f"query: {text}"])[0], dtype=np.float32)


class HashEmbedder:
    """Bag-of-stems hashed into a fixed vector. No model download; lexical only. For tests and CI."""

    def __init__(self, dim: int = 256) -> None:
        self.name = f"hash-{dim}"
        self.dim = dim

    def _one(self, text: str) -> Vector:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=4).digest(), "little")
            v[h % self.dim] += 1.0
        n = float(np.linalg.norm(v))
        return v / n if n else v

    def embed_documents(self, texts: list[str]) -> Vector:
        return np.stack([self._one(t) for t in texts]) if texts else np.zeros((0, self.dim), dtype=np.float32)

    def embed_query(self, text: str) -> Vector:
        return self._one(text)


@lru_cache(maxsize=2)
def _load(backend: str, model_name: str) -> Embedder:
    if backend == "onnx":
        return OnnxE5Embedder(model_name)
    return E5Embedder(model_name)


def get_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "hash":
        return HashEmbedder()
    return _load(settings.embedding_backend, settings.embedding_model)
