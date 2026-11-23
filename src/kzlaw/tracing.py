"""Langfuse tracing. Off unless LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set.

One trace per question: the root "agent" observation holds input/output; LLM calls, tool calls and
their latency/tokens come from the LangChain callback handler nested under it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from kzlaw.config import Settings


@dataclass
class TraceContext:
    callbacks: list[Any] = field(default_factory=list)
    trace_id: str | None = None
    output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@lru_cache(maxsize=1)
def _client(public_key: str, secret_key: str, host: str) -> Any:
    from langfuse import Langfuse

    return Langfuse(public_key=public_key, secret_key=secret_key, base_url=host)


def get_langfuse(settings: Settings) -> Any | None:
    if not settings.tracing_enabled:
        return None
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return None
    return _client(
        settings.langfuse_public_key, settings.langfuse_secret_key.get_secret_value(), settings.langfuse_host
    )


@contextmanager
def trace(settings: Settings, name: str, input: Any, metadata: dict[str, Any] | None = None) -> Iterator[TraceContext]:
    lf = get_langfuse(settings)
    if lf is None:
        yield TraceContext()
        return
    from langfuse.langchain import CallbackHandler

    with lf.start_as_current_observation(name=name, as_type="agent", input=input, metadata=metadata or {}) as obs:
        ctx = TraceContext(callbacks=[CallbackHandler(public_key=settings.langfuse_public_key)])
        ctx.trace_id = lf.get_current_trace_id()
        yield ctx
        obs.update(output=ctx.output, metadata={**(metadata or {}), **ctx.metadata})


def flush(settings: Settings) -> None:
    lf = get_langfuse(settings)
    if lf is not None:
        lf.flush()
