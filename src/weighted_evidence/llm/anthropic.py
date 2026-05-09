"""Anthropic provider with prompt caching on the system prompt."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from weighted_evidence.config import Settings
from weighted_evidence.config import settings as load_settings
from weighted_evidence.llm.base import LLMResponse

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic  # noqa: F401  (typing only)


class AnthropicProvider:
    name: str = "anthropic"

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        model: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.model = model or self.settings.default_llm_model
        self._client = client

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        from anthropic import AsyncAnthropic

        if self.settings.anthropic_api_key is None:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it or pass an explicit client."
            )
        self._client = AsyncAnthropic(api_key=self.settings.anthropic_api_key)
        return self._client

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
    ) -> LLMResponse:
        client = self._ensure_client()
        system_blocks = [
            {
                "type": "text",
                "text": system,
                **({"cache_control": {"type": "ephemeral"}} if cacheable_system else {}),
            }
        ]
        tools = [
            {
                "name": tool_name,
                "description": tool_description,
                "input_schema": tool_schema,
            }
        ]
        msg = await client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_blocks,
            tools=tools,
            tool_choice={"type": "tool", "name": tool_name},
            messages=[{"role": "user", "content": user}],
        )
        content: dict[str, Any] = {}
        for block in msg.content:
            if getattr(block, "type", None) == "tool_use":
                content = cast(dict[str, Any], block.input)
                break

        return LLMResponse(
            content=content,
            raw_text=None,
            model=getattr(msg, "model", self.model),
            cache_hit=bool(getattr(getattr(msg, "usage", None), "cache_read_input_tokens", 0) or 0),
        )
