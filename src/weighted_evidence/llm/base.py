"""LLM provider Protocol and shared response shape."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """Provider-agnostic response shape for structured tool-use extraction."""

    content: dict[str, Any] = Field(default_factory=dict)
    raw_text: str | None = None
    model: str | None = None
    cache_hit: bool = False


@runtime_checkable
class LLMProvider(Protocol):
    """All providers MUST surface tool-use style structured output.

    Calls return a single `LLMResponse` whose `content` dict matches the supplied
    `tool_schema`. `cacheable_blocks` lets callers mark static system prompts as
    cacheable for prompt caching; providers that don't support caching ignore it.
    """

    name: str

    async def structured(
        self,
        *,
        system: str,
        user: str,
        tool_name: str,
        tool_description: str,
        tool_schema: dict[str, Any],
        cacheable_system: bool = True,
        max_tokens: int = 2048,
    ) -> LLMResponse: ...
