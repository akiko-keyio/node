"""Node - A lightweight DAG engine with caching and parallel execution.

Node is a lightweight Directed Acyclic Graph (DAG) computation framework with automatic caching and parallel execution capabilities.

Quick Start
-----------
>>> import node
>>> 
>>> # Define a computation node
>>> @node.define()
... def add(x: int, y: int) -> int:
...     return x + y
>>> 
>>> # Build and execute DAG
>>> result = add(1, 2).get()
>>> print(result)
3

Configuration
-------------
>>> node.configure(workers=4, cache_root="./cache")

See Also
--------
node.define : Decorate a function to create a Node.
node.run : Execute the entire DAG.
node.gather : Aggregate multiple nodes.
"""

from collections.abc import Callable, Generator
from typing import Any

# Core exports
from .core import Node, gather, sweep
from .cache import Cache, ChainCache, MemoryLRU, DiskJoblib
from .config import Config
from .runtime import Runtime, get_runtime, configure, reset
from .logger import logger, console

# Module-level API that delegates to singleton Runtime
def run(root: Node, *, reporter=None, cache_root: bool = True):
    """Run the DAG rooted at ``root``.
    
    Parameters
    ----------
    root : Node
        The solution node (root of the DAG) to execute.
    reporter : Reporter, optional
        Progress reporter instance.
    cache_root : bool, optional
        Whether to cache the result of the root node itself. 
        Defaults to True.

    Returns
    -------
    Any
        The result of the computation.

    Examples
    --------
    >>> result = node.run(my_task(1, 2))
    """
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
    ignore : list[str], optional
        Argument names excluded from the cache key.
    workers : int, optional
        Maximum concurrency for this function. ``-1`` uses all cores.
    cache : bool, optional
        Whether to cache the result. Defaults to True.
    local : bool, optional
        Execute directly in the caller thread, bypassing any executor. 
        Defaults to False.

    Returns
    -------
    Callable
        A decorator that converts the function into a Node factory.

    Examples
    --------
    >>> @node.define(workers=2)
    ... def slow_task(x):
    ...     time.sleep(1)
    ...     return x
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



# Optional rich dependency
track: Callable[..., Generator[Any, None, None]] | None = None
RichReporter: type | None = None

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
    "sweep",
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
    # Optional
    "RichReporter",
    "track",
    "logger",
    "console",
]
