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
    """Thread-safe in-memory LRU cache."""

    def __init__(self, maxsize: int = 512):
        self._lru: LRUCache[int, Any] = LRUCache(maxsize=maxsize)
        self._lock = threading.Lock()

    def get(self, fn_name: str, hash_value: int):
        with self._lock:
            if hash_value in self._lru:
                return True, self._lru[hash_value]
        return False, None

    def put(self, fn_name: str, hash_value: int, value: Any):
        with self._lock:
            self._lru[hash_value] = value

    def delete(self, fn_name: str, hash_value: int) -> None:
        with self._lock:
            self._lru.pop(hash_value, None)


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

    def _path(self, fn_name: str, hash_value: int, ext: str = ".pkl") -> Path:
        """Return the cache file path.
        
        Args:
            fn_name: Function name.
            hash_value: Node hash value.
            ext: File extension.
            
        Returns:
            Path to the cache file.
        """
        sub = self.root / fn_name
        sub.mkdir(parents=True, exist_ok=True)
        return sub / (f"{hash_value:x}" + ext)

    def get(self, fn_name: str, hash_value: int):
        p = self._path(fn_name, hash_value)
        
        try:
            if p.stat().st_size <= self.small_file:
                try:
                    with p.open("rb") as fh:
                        return True, pickle.load(fh)
                except FileNotFoundError:
                    return False, None
            
            # joblib.load raises FileNotFoundError if file is missing
            return True, joblib.load(p)

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
        return False, None

    def put(self, fn_name: str, hash_value: int, value: Any):
        p = self._path(fn_name, hash_value)
        lock_path = str(p) + ".lock"
        ctx = FileLock(lock_path) if self.lock else nullcontext()
        with ctx:
            data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            if len(data) <= self.small_file:
                with p.open("wb") as fh:
                    fh.write(data)
            else:
                joblib.dump(value, p)

    def delete(self, fn_name: str, hash_value: int) -> None:
        for ext in (".pkl", ".py"):
            p = self._path(fn_name, hash_value, ext)
            if p.exists():
                with suppress(OSError):
                    p.unlink()

    def save_script(self, node: "Node"):
        p = self._path(node.fn.__name__, node._hash, ".py")
        p.write_text(repr(node) + "\n")


class ChainCache(Cache):
    """Chain several caches (e.g. Memory → Disk)."""

    def __init__(self, caches: list[Cache]):
        self.caches = list(caches)
        self._lock = threading.Lock()

    def get(self, fn_name: str, hash_value: int):
        with self._lock:
            for i, c in enumerate(self.caches):
                hit, val = c.get(fn_name, hash_value)
                if hit:
                    for earlier in self.caches[:i]:
                        earlier.put(fn_name, hash_value, val)
                    return True, val
        return False, None

    def put(self, fn_name: str, hash_value: int, value: Any):
        for c in self.caches:
            c.put(fn_name, hash_value, value)

    def delete(self, fn_name: str, hash_value: int) -> None:
        for c in self.caches:
            c.delete(fn_name, hash_value)

    def save_script(self, node: "Node"):
        for c in self.caches:
            c.save_script(node)
