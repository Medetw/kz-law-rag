"""Settings from environment variables / .env (same LLM naming as the author's other projects)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://kzlaw:kzlaw@localhost:5433/kzlaw"
    raw_dir: Path = Path("data/raw")

    # retrieval
    # onnx = int8 E5 on onnxruntime (default, no torch); e5 = full-precision E5 via sentence-transformers;
    # hash = dependency-free lexical embedder for tests / CI
    embedding_backend: Literal["onnx", "e5", "hash"] = "onnx"
    embedding_model: str = "intfloat/multilingual-e5-small"
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    rerank: bool = False

    # LLM (OpenAI-compatible chat API)
    llm_provider: Literal["deepseek", "anthropic"] = "deepseek"
    llm_model: str | None = None
    deepseek_api_key: SecretStr | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    anthropic_api_key: SecretStr | None = None
    anthropic_base_url: str = "https://api.anthropic.com/v1/"
    llm_temperature: float = 0.0

    # observability (Langfuse); tracing is off unless both keys are set
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com", validation_alias=AliasChoices("LANGFUSE_BASE_URL", "LANGFUSE_HOST")
    )

    @property
    def resolved_llm_model(self) -> str:
        return self.llm_model or ("deepseek-flash" if self.llm_provider == "deepseek" else "claude-haiku-5-5")

    @property
    def tracing_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
