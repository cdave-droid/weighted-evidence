"""Retrieval modules: resolve identifiers and fetch normalized Paper records."""

from weighted_evidence.retrieval.client import RetrievalClient
from weighted_evidence.retrieval.resolver import IdentifierKind, resolve

__all__ = ["IdentifierKind", "RetrievalClient", "resolve"]
