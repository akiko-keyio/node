from typing import Optional, Callable, Generator, Any

from .node import (
    ChainCache,
    Config,
    DiskJoblib,
    Flow,
    MemoryLRU,
    Node,
    gather,
)
from .logger import logger, console

track: Optional[Callable[..., Generator[Any, None, None]]] = None

try:  # optional rich dependency
    from .reporters import RichReporter as _RichReporter, track as _track

    RichReporter: Optional[type] = _RichReporter
    track = _track
except Exception:  # pragma: no cover - optional
    RichReporter = None

__all__ = [
    "Node",
    "ChainCache",
    "MemoryLRU",
    "DiskJoblib",
    "Config",
    "Flow",
    "gather",
    "RichReporter",
    "track",
    "logger",
    "console",
]
