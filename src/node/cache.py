"""Cache abstractions for result storage."""

from __future__ import annotations

import pickle
import threading
from contextlib import nullcontext, suppress
from pathlib import Path
from typing import Any, TYPE_CHECKING

from cachetools import LRUCache  # type: ignore[import]
from filelock import FileLock  # type: ignore[import]

from .cache_namespace import cache_namespace
from .logger import logger

if TYPE_CHECKING:
    from .core import Node

__all__ = [
    "Cache",
    "MemoryLRU",
    "DiskCache",
    "ChainCache",
]


class Cache:
    """Base cache interface."""

    def _has_entry(self, fn_name: str, hash_value: int) -> bool:
        """Internal lightweight cache-hit check used by graph pruning."""
        hit, _ = self.get(fn_name, hash_value)
        return hit

    def get(self, fn_name: str, hash_value: int) -> tuple[bool, Any]:
        """Get a cached value.
        
        Args:
            fn_name: Function name.
            hash_value: Node hash value.
            
        Returns:
            A tuple of (hit, value).
        """
        raise NotImplementedError

    def put(self, fn_name: str, hash_value: int, value: Any) -> None:
        """Store a value in the cache.
        
        Args:
            fn_name: Function name.
            hash_value: Node hash value.
            value: Value to cache.
        """
        raise NotImplementedError

    def delete(self, fn_name: str, hash_value: int) -> None:
        """Delete a cached value.
        
        Args:
            fn_name: Function name.
            hash_value: Node hash value.
        """
        raise NotImplementedError

    def save_script(self, node: "Node") -> None:
        """Save the node's script representation."""
        pass


class MemoryLRU(Cache):
    """In-memory LRU cache."""

    def __init__(self, maxsize: int = 512):
        self._lru: LRUCache[int, Any] = LRUCache(maxsize=maxsize)

    def get(self, fn_name: str, hash_value: int):
        try:
            return True, self._lru[hash_value]
        except KeyError:
            return False, None

    def put(self, fn_name: str, hash_value: int, value: Any):
        self._lru[hash_value] = value

    def delete(self, fn_name: str, hash_value: int) -> None:
        self._lru.pop(hash_value, None)


class DiskCache(Cache):
    """Filesystem cache using pickle.

    Results are stored under ``<func>/<hash>.pkl``. The corresponding script
    is written alongside each entry as ``.py`` for inspection.

    Cache payloads are serialized with ``pickle`` and stored as ``.pkl`` files.
    """

    def __init__(
        self,
        root: str | Path = ".cache",
        lock: bool = True,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = lock
        self._index: dict[str, set[int]] = {}
        self._index_lock = threading.Lock()

    def _path(self, fn_name: str, hash_value: int, ext: str = ".pkl") -> Path:
        """Return the cache file path.
        
        Args:
            fn_name: Function name.
            hash_value: Node hash value.
            ext: File extension.
            
        Returns:
            Path to the cache file.
        """
        return (self.root / fn_name) / (f"{hash_value:x}" + ext)

    @staticmethod
    def _normalize_namespace(fn_name: str) -> str:
        return fn_name.replace("\\", "/")

    def _load_namespace_index(self, fn_name: str) -> set[int]:
        ns = self._normalize_namespace(fn_name)
        with self._index_lock:
            cached = self._index.get(ns)
            if cached is not None:
                return cached

            entries: set[int] = set()
            namespace_dir = self.root / ns
            try:
                for path in namespace_dir.glob("*.pkl"):
                    try:
                        entries.add(int(path.stem, 16))
                    except ValueError:
                        continue
            except OSError:
                entries = set()
            self._index[ns] = entries
            return entries

    def _index_add(self, fn_name: str, hash_value: int) -> None:
        ns = self._normalize_namespace(fn_name)
        with self._index_lock:
            if ns in self._index:
                self._index[ns].add(hash_value)

    def _index_discard(self, fn_name: str, hash_value: int) -> None:
        ns = self._normalize_namespace(fn_name)
        with self._index_lock:
            if ns in self._index:
                self._index[ns].discard(hash_value)

    def _has_entry(self, fn_name: str, hash_value: int) -> bool:
        return hash_value in self._load_namespace_index(fn_name)

    def get(self, fn_name: str, hash_value: int):
        p = self._path(fn_name, hash_value)

        try:
            with p.open("rb") as fh:
                return True, pickle.load(fh)
        except FileNotFoundError:
            return False, None
        except Exception as exc:  # pragma: no cover - defensive
            if isinstance(
                exc, (pickle.UnpicklingError, EOFError, AttributeError, ValueError)
            ):
                return self._handle_corrupt_cache(p, exc)
            from .exceptions import CacheError
            raise CacheError(f"Failed to load cache file {p}: {exc}") from exc
            

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
        try:
            self._index_discard(
                self._normalize_namespace(str(path.parent.relative_to(self.root))),
                int(path.stem, 16),
            )
        except (ValueError, OSError):
            pass
        return False, None

    def put(self, fn_name: str, hash_value: int, value: Any):
        p = self._path(fn_name, hash_value)
        p.parent.mkdir(parents=True, exist_ok=True)
        lock_path = str(p) + ".lock"
        ctx = FileLock(lock_path) if self.lock else nullcontext()
        with ctx:
            with p.open("wb") as fh:
                pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)
        self._index_add(fn_name, hash_value)

    def delete(self, fn_name: str, hash_value: int) -> None:
        for ext in (".pkl", ".py"):
            p = self._path(fn_name, hash_value, ext)
            if p.exists():
                with suppress(OSError):
                    p.unlink()
        self._index_discard(fn_name, hash_value)

    def save_script(self, node: "Node"):
        p = self._path(cache_namespace(node), node._hash, ".py")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(repr(node) + "\n")


class ChainCache(Cache):
    """Chain several caches (e.g. Memory → Disk)."""

    def __init__(self, caches: list[Cache]):
        self.caches = list(caches)
        self._lock = threading.Lock()

    def get(self, fn_name: str, hash_value: int):
        if not self.caches:
            return False, None

        # Fast path: in hot runs most lookups hit L1 cache.
        # Avoid a global lock here to reduce contention in threaded execution.
        hit, val = self.caches[0].get(fn_name, hash_value)
        if hit:
            return True, val

        # Slow path: when a lower layer hits, serialize promotion to keep
        # behavior deterministic and avoid duplicate put storms.
        with self._lock:
            # Re-check L1 after waiting for the lock: another thread may have
            # already promoted this key.
            hit, val = self.caches[0].get(fn_name, hash_value)
            if hit:
                return True, val

            for i, c in enumerate(self.caches[1:], start=1):
                hit, val = c.get(fn_name, hash_value)
                if hit:
                    for earlier in self.caches[:i]:
                        earlier.put(fn_name, hash_value, val)
                    return True, val
        return False, None

    def _has_entry(self, fn_name: str, hash_value: int) -> bool:
        if not self.caches:
            return False
        for c in self.caches:
            if c._has_entry(fn_name, hash_value):
                return True
        return False

    def put(self, fn_name: str, hash_value: int, value: Any):
        for c in self.caches:
            c.put(fn_name, hash_value, value)

    def delete(self, fn_name: str, hash_value: int) -> None:
        for c in self.caches:
            c.delete(fn_name, hash_value)

    def save_script(self, node: "Node"):
        for c in self.caches:
            c.save_script(node)
