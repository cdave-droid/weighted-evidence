"""Helpers that wrap a provider call into a `Tracked[T]` value."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from weighted_evidence.llm.base import LLMProvider
from weighted_evidence.models import Provenance, Tracked

ModelT = TypeVar("ModelT", bound=BaseModel)


class StructuredCall(Generic[ModelT]):
    """Bind a Pydantic model to a provider call, returning Tracked[ModelT]."""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        model_cls: type[ModelT],
        tool_name: str,
        tool_description: str,
        tool_schema: dict[str, Any],
    ) -> None:
        self.provider = provider
        self.model_cls = model_cls
        self.tool_name = tool_name
        self.tool_description = tool_description
        self.tool_schema = tool_schema

    async def __call__(
        self,
        *,
        system: str,
        user: str,
        provenance: Provenance | None = None,
        cacheable_system: bool = True,
    ) -> Tracked[ModelT]:
        response = await self.provider.structured(
            system=system,
            user=user,
            tool_name=self.tool_name,
            tool_description=self.tool_description,
            tool_schema=self.tool_schema,
            cacheable_system=cacheable_system,
        )
        confidence = 1.0
        if isinstance(response.content, dict) and "confidence" in response.content:
            try:
                confidence = float(response.content.pop("confidence"))
            except (TypeError, ValueError):
                confidence = 1.0
        value = self.model_cls.model_validate(response.content)
        return Tracked[ModelT](
            value=value,
            confidence=max(0.0, min(1.0, confidence)),
            provenance=provenance,
            extractor="llm",
        )
