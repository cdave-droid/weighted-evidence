"""SQLite-backed cache. Namespaced (`http`, `llm`, `report`, `retraction`) with per-namespace TTL."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, ClassVar

import sqlite_utils
from sqlite_utils.db import NotFoundError, Table


class Cache:
    DEFAULT_TTL: ClassVar[dict[str, int]] = {
        "http": 60 * 60 * 24 * 7,  # 7 days
        "llm": 60 * 60 * 24 * 30,  # 30 days (deterministic per model_version)
        "report": 60 * 60 * 24 * 30,
        "retraction": 60 * 60 * 24,  # 1 day — retraction status changes
    }

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite_utils.Database(str(path))
        self._ensure_schema()
        self._kv: Table = self.db.table("kv")  # type: ignore[assignment]

    def _ensure_schema(self) -> None:
        if "kv" not in self.db.table_names():
            self.db.table("kv").create(  # type: ignore[union-attr]
                {
                    "namespace": str,
                    "key": str,
                    "value": str,
                    "stored_at": int,
                    "ttl": int,
                },
                pk=("namespace", "key"),
            )

    def get(self, namespace: str, key: str) -> str | None:
        try:
            row = self._kv.get((namespace, key))
        except NotFoundError:
            return None
        if int(time.time()) - row["stored_at"] > row["ttl"]:
            return None
        return str(row["value"])

    def set(
        self,
        namespace: str,
        key: str,
        value: str,
        ttl: int | None = None,
    ) -> None:
        ttl = ttl if ttl is not None else self.DEFAULT_TTL.get(namespace, 60 * 60 * 24)
        self._kv.upsert(
            {
                "namespace": namespace,
                "key": key,
                "value": value,
                "stored_at": int(time.time()),
                "ttl": ttl,
            },
            pk=("namespace", "key"),
        )

    def clear(self, namespace: str | None = None) -> int:
        if namespace is None:
            count = self._kv.count
            self._kv.delete_where()
            return count
        rows = list(self._kv.rows_where("namespace = ?", [namespace]))
        self._kv.delete_where("namespace = ?", [namespace])
        return len(rows)


def open_cache(path: Path | None = None) -> Cache:
    from weighted_evidence.config import settings

    target = path or (settings().cache_dir / "cache.sqlite")
    return Cache(Path(target))


def cache_key(*parts: Any) -> str:
    """Stable cache key from arbitrary parts."""

    return "|".join(str(p) for p in parts)
