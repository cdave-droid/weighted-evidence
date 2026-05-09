"""Settings loaded from environment / .env."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WEIGHTED_EVIDENCE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    ncbi_api_key: str | None = Field(default=None, alias="NCBI_API_KEY")
    semantic_scholar_api_key: str | None = Field(default=None, alias="SEMANTIC_SCHOLAR_API_KEY")

    cache_dir: Path = Field(default=Path(".weighted_evidence_cache"))
    default_llm_model: str = Field(default="claude-opus-4-7")
    pubmed_tool: str = Field(default="weighted-evidence")
    pubmed_email: str | None = Field(default=None)


_cached: Settings | None = None


def settings() -> Settings:
    global _cached
    if _cached is None:
        _cached = Settings()
    return _cached
