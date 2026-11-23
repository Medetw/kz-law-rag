"""Chat model factory: DeepSeek or Claude, both through their OpenAI-compatible endpoints."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from kzlaw.config import Settings


class MissingKeyError(RuntimeError):
    pass


def get_chat_model(settings: Settings) -> ChatOpenAI:
    if settings.llm_provider == "deepseek":
        key, base_url = settings.deepseek_api_key, settings.deepseek_base_url
    else:
        key, base_url = settings.anthropic_api_key, settings.anthropic_base_url
    if key is None:
        env = "DEEPSEEK_API_KEY" if settings.llm_provider == "deepseek" else "ANTHROPIC_API_KEY"
        raise MissingKeyError(f"LLM_PROVIDER={settings.llm_provider} needs {env} in the environment or .env")
    return ChatOpenAI(
        model=settings.resolved_llm_model,
        api_key=key,
        base_url=base_url,
        temperature=settings.llm_temperature,
        timeout=90,
        max_retries=2,
    )
