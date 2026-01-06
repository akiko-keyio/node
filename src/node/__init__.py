"""Node - A lightweight DAG engine with caching and parallel execution.

Usage:
    import node

    node.configure(workers=4)  # Optional, uses defaults if not called

    @node.define
    def my_task(x, y):
        return x + y

    result = node.run(my_task(1, 2))
"""

from typing import Optional, Callable, Generator, Any

# Core exports
from .core import Node, gather, map
from .cache import Cache, ChainCache, MemoryLRU, DiskJoblib
from .config import Config
from .runtime import Runtime, get_runtime, configure, reset
from .logger import logger, console

# Module-level API that delegates to singleton Runtime
def run(root: Node, *, reporter=None, cache_root: bool = True):
    """Run the DAG rooted at ``root``."""
    return get_runtime().run(root, reporter=reporter, cache_root=cache_root)


def define(
    *,
    ignore: list[str] | None = None,
    workers: int | None = None,
    cache: bool = True,
    local: bool = False,
):
    """Decorate a function to create a Node.

    Parameters
    ----------
    ignore:
        Argument names excluded from the cache key.
    workers:
        Maximum concurrency for this function. ``-1`` uses all cores.
    cache:
        Whether to cache the result.
    local:
        Execute directly in the caller thread, bypassing any executor.
    """
    return get_runtime().define(ignore=ignore, workers=workers, cache=cache, local=local)


class _CfgProxy:
    """Proxy object for convenient access to runtime configuration.
    
    Allows `node.cfg.xxx` syntax instead of `node.get_runtime().config._conf.xxx`.
    """
    
    def __getattr__(self, name: str):
        return getattr(get_runtime().config._conf, name)
    
    def __setattr__(self, name: str, value) -> None:
        setattr(get_runtime().config._conf, name, value)
    
    def __repr__(self) -> str:
        return repr(get_runtime().config._conf)


# Module-level config proxy
cfg = _CfgProxy()

# Backwards compatibility
Flow = Runtime

# Optional rich dependency
track: Optional[Callable[..., Generator[Any, None, None]]] = None
RichReporter: Optional[type] = None

try:
    from .reporters import RichReporter as _RichReporter, track as _track

    RichReporter = _RichReporter
    track = _track
except Exception:  # pragma: no cover - optional
    pass

__all__ = [
    # New API
    "configure",
    "define",
    "run",
    "gather",
    "map",
    "reset",
    # Core classes
    "Node",
    "Runtime",
    # Cache
    "Cache",
    "ChainCache",
    "MemoryLRU",
    "DiskJoblib",
    # Config
    "Config",
    "cfg",
    # Backwards compatibility
    "Flow",
    # Optional
    "RichReporter",
    "track",
    "logger",
    "console",
]
