"""Cache abstractions for result storage."""

from __future__ import annotations

import pickle
import threading
from contextlib import nullcontext, suppress
from pathlib import Path
from typing import Any, TYPE_CHECKING

import joblib  # type: ignore[import]
from cachetools import LRUCache  # type: ignore[import]
from filelock import FileLock  # type: ignore[import]

from .logger import logger

if TYPE_CHECKING:
    from .core import Node

__all__ = [
    "Cache",
    "MemoryLRU",
    "DiskJoblib",
    "ChainCache",
]


class Cache:
    """Base cache interface."""

    def get(self, key: str) -> tuple[bool, Any]:
        raise NotImplementedError

    def put(self, key: str, value: Any) -> None:
        raise NotImplementedError

    def delete(self, key: str) -> None:
        raise NotImplementedError

    def save_script(self, node: "Node") -> None:
        pass


class MemoryLRU(Cache):
    """Thread-safe in-memory LRU cache."""

    def __init__(self, maxsize: int = 512):
        self._lru: LRUCache[str, Any] = LRUCache(maxsize=maxsize)
        self._lock = threading.Lock()

    def get(self, key: str):
        with self._lock:
            if key in self._lru:
                return True, self._lru[key]
        return False, None

    def put(self, key: str, value: Any):
        with self._lock:
            self._lru[key] = value

    def delete(self, key: str) -> None:
        with self._lock:
            self._lru.pop(key, None)


class DiskJoblib(Cache):
    """Filesystem cache using joblib pickles.

    Results are stored under ``<func>/<hash>.pkl`` and the corresponding script
    is written to ``<func>/<hash>.py`` for inspection.

    ``small_file`` sets a byte threshold below which ``pickle`` is used for
    loading to avoid ``joblib`` overhead on many tiny files.
    """

    def __init__(
        self,
        root: str | Path = ".cache",
        lock: bool = True,
        *,
        small_file: int = 1_000_000,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = lock
        self.small_file = small_file

    def _path(self, key: str, ext: str = ".pkl") -> Path:
        """Return the cache file path for ``key``."""
        fn, hash_key = key.rsplit("_", 1)
        sub = self.root / fn
        sub.mkdir(parents=True, exist_ok=True)
        return sub / (hash_key + ext)

    def get(self, key: str):
        p = self._path(key)
        if not p.exists():
            return False, None

        try:
            if p.stat().st_size <= self.small_file:
                with p.open("rb") as fh:
                    return True, pickle.load(fh)
            return True, joblib.load(p)
        except Exception as exc:  # pragma: no cover - defensive
            if isinstance(
                exc, (pickle.UnpicklingError, EOFError, AttributeError, ValueError)
            ):
                return self._handle_corrupt_cache(p, exc)
            raise RuntimeError(f"Failed to load cache file {p}: {exc}") from exc

    def _handle_corrupt_cache(self, path: Path, error: Exception) -> tuple[bool, None]:
        logger.error(
            "Failed to load cache file %s: %s. Deleting the cache entry.",
            path,
            error,
        )
        with suppress(OSError):
            path.unlink()
        with suppress(OSError):
            path.with_suffix(".py").unlink()
        return False, None

    def put(self, key: str, value: Any):
        p = self._path(key)
        lock_path = str(p) + ".lock"
        ctx = FileLock(lock_path) if self.lock else nullcontext()
        with ctx:
            data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            if len(data) <= self.small_file:
                with p.open("wb") as fh:
                    fh.write(data)
            else:
                joblib.dump(value, p)

    def delete(self, key: str) -> None:
        for ext in (".pkl", ".py"):
            p = self._path(key, ext)
            if p.exists():
                with suppress(OSError):
                    p.unlink()

    def save_script(self, node: "Node"):
        p = self._path(node.key, ".py")
        p.write_text(repr(node) + "\n")


class ChainCache(Cache):
    """Chain several caches (e.g. Memory → Disk)."""

    def __init__(self, caches: list[Cache]):
        self.caches = list(caches)
        self._lock = threading.Lock()

    def get(self, key: str):
        with self._lock:
            for i, c in enumerate(self.caches):
                hit, val = c.get(key)
                if hit:
                    for earlier in self.caches[:i]:
                        earlier.put(key, val)
                    return True, val
        return False, None

    def put(self, key: str, value: Any):
        for c in self.caches:
            c.put(key, value)

    def delete(self, key: str) -> None:
        for c in self.caches:
            c.delete(key)

    def save_script(self, node: "Node"):
        for c in self.caches:
            c.save_script(node)
