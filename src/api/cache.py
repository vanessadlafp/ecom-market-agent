"""Simple in-memory TTL cache. Swap for Redis in production."""

from __future__ import annotations
import time
from typing import Any
from config.settings import get_settings


class _Entry:
    __slots__ = ("value", "expires_at")
    def __init__(self, value: Any, ttl: float):
        self.value = value
        self.expires_at = time.monotonic() + ttl


class MemoryCache:
    def __init__(self, ttl: int | None = None):
        self._store: dict[str, _Entry] = {}
        self._ttl = ttl or get_settings().cache_ttl_seconds

    def get(self, key: str) -> Any | None:
        e = self._store.get(key)
        if e is None or time.monotonic() > e.expires_at:
            self._store.pop(key, None)
            return None
        return e.value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        self._store[key] = _Entry(value, ttl or self._ttl)

    def clear(self) -> None:
        self._store.clear()


_cache: MemoryCache | None = None

def get_cache() -> MemoryCache:
    global _cache
    if _cache is None:
        _cache = MemoryCache()
    return _cache
