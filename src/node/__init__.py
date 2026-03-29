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
>>> result = add(1, 2)()
>>> print(result)
3

Configuration
-------------
>>> node.configure(workers=4, cache_root="./cache")

See Also
--------
node.define : Decorate a function to create a Node.
"""

from collections.abc import Callable, Generator, Mapping
from typing import Any

# Core exports
from .core import Node, dimension, define
from .exceptions import NodeError, ConfigurationError, DimensionMismatchError, CacheError
from .cache import Cache, ChainCache, MemoryLRU, DiskCache
from .config import Config
from .runtime import Runtime, get_runtime, configure, reset
from .logger import logger, console


def show_logs() -> None:
    """Print buffered log messages from the last notebook execution.

    In Jupyter, log output is suppressed during live progress display to
    prevent conflicts.  Call ``node.show_logs()`` in a separate cell to
    review those messages afterwards.
    """
    rt = get_runtime()
    logs = getattr(rt.reporter, "_last_logs", None) or getattr(
        getattr(rt.reporter, "cfg", None), "_last_logs", None
    )
    if not logs:
        print("(no buffered logs)")
        return
    for line in logs:
        print(line, end="")


def instantiate(name: str, *, sweep: Mapping[str, Any] | None = None) -> Node:
    """Instantiate a node from current runtime config by section name.

    The config is read when this function is called. The returned node keeps that
    bound configuration; later ``node.cfg`` updates do not retroactively change it.
    Re-run ``instantiate()`` after config edits if you want a new bound node.
    When ``sweep`` is provided, instantiate returns a dimensioned node over the
    Cartesian product of sweep values. Sweep keys use global config paths such as
    ``"train.optimizer"`` or ``"trop_ls.degree"``.
    """
    runtime = get_runtime()
    return runtime.config.instantiate(name, runtime=runtime, sweep=sweep)


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
    from .reporter import RichReporter as _RichReporter, track as _track

    RichReporter = _RichReporter
    track = _track
except Exception:  # pragma: no cover - optional
    pass

__all__ = [
    # Core API
    "configure",
    "define",
    "instantiate",
    "dimension",
    "reset",
    # Exceptions
    "NodeError",
    "ConfigurationError",
    "DimensionMismatchError",
    "CacheError",
    # Core classes
    "Node",
    "Runtime",
    # Cache
    "Cache",
    "ChainCache",
    "MemoryLRU",
    "DiskCache",
    # Config
    "Config",
    "cfg",
    "show_logs",
    # Optional
    "RichReporter",
    "track",
    "logger",
    "console",
]
