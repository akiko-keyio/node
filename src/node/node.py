"""Minimal DAG engine with in-memory and disk caching."""

from __future__ import annotations

import hashlib
import inspect
import functools
import os
import threading
import time
import warnings
from collections import deque
from collections.abc import Mapping, Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from contextlib import nullcontext, suppress
from graphlib import TopologicalSorter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from weakref import WeakValueDictionary

import joblib  # type: ignore[import]
from cachetools import LRUCache  # type: ignore[import]
from filelock import FileLock  # type: ignore[import]

__all__ = [
    "Cache",
    "MemoryLRU",
    "DiskJoblib",
    "ChainCache",
    "Node",
    "Config",
    "Flow",
]

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from .reporters import RichReporter  # noqa: F401


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

# global caches & locks
_can_lock = threading.Lock()
_ren_lock = threading.Lock()
_canonical_cache: LRUCache[tuple[int, str], str] = LRUCache(maxsize=4096)
_render_cache: LRUCache[tuple, str] = LRUCache(maxsize=2048)


def _canonical(obj: Any) -> str:
    """Convert an object into a deterministic string representation."""
    if isinstance(obj, Node):
        return obj.key

    key = (id(obj), repr(obj))
    with _can_lock:
        res = _canonical_cache.get(key)
    if res is not None:
        return res

    if isinstance(obj, dict):
        inner = ", ".join(f"{repr(k)}: {_canonical(v)}" for k, v in sorted(obj.items()))
        res = "{" + inner + "}"
    elif isinstance(obj, (list, tuple)):
        inner = ", ".join(_canonical(v) for v in obj)
        res = "[" + inner + "]" if isinstance(obj, list) else "(" + inner + ")"
    elif isinstance(obj, set):
        inner = ", ".join(_canonical(v) for v in sorted(obj))
        res = "{" + inner + "}"
    else:
        res = repr(obj)

    with _can_lock:
        _canonical_cache[key] = res
    return res


def _canonical_args(node: "Node") -> Tuple[Tuple[str, str], ...]:
    sig = getattr(node.fn, "_node_sig", None)
    if sig is None:
        sig = inspect.signature(node.fn)
        setattr(node.fn, "_node_sig", sig)
    bound = sig.bind_partial(*node.args, **node.kwargs)
    ignore = set(getattr(node.fn, "_node_ignore", ()))
    parts = []
    for k, v in bound.arguments.items():
        if k in ignore:
            continue
        if isinstance(v, Node):
            parts.append((k, f"{v._hash:032x}"))
        else:
            parts.append((k, _canonical(v)))
    return tuple(parts)


# ----------------------------------------------------------------------
# cache abstractions
# ----------------------------------------------------------------------
class Cache:
    def get(self, key: str) -> Tuple[bool, Any]:
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
    """

    def __init__(self, root: str | Path = ".cache", lock: bool = True):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = lock

    def _path(self, key: str, ext: str = ".pkl") -> Path:
        """Return the cache file path for ``key``."""
        parts = key.rsplit("_", 1)
        fn = parts[0]
        hash_key = parts[1]
        sub = self.root / fn
        sub.mkdir(parents=True, exist_ok=True)
        return sub / (hash_key + ext)

    def get(self, key: str):
        p = self._path(key)
        if p.exists():
            return True, joblib.load(p)
        return False, None

    def put(self, key: str, value: Any):
        p = self._path(key)
        lock_path = str(p) + ".lock"
        ctx = FileLock(lock_path) if self.lock else nullcontext()
        with ctx:
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

    def __init__(self, caches: Sequence[Cache]):
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


# ----------------------------------------------------------------------
# DAG nodes
# ----------------------------------------------------------------------
class Node:
    __slots__ = (
        "fn",
        "args",
        "kwargs",
        "deps",
        "flow",
        "_hash",
        "_raw",
        "_lock",
        "_ancestors",
        "__dict__",
        "__weakref__",
    )

    _hash: int
    _lock: threading.Lock

    def __init__(
        self,
        fn,
        args: Tuple = (),
        kwargs: Dict | None = None,
        *,
        flow: "Flow" | None = None,
    ):
        self.fn = fn
        self.args = tuple(args)
        self.kwargs = kwargs or {}
        self.flow = flow

        self.deps: List[Node] = [
            *(a for a in self.args if isinstance(a, Node)),
            *(v for v in self.kwargs.values() if isinstance(v, Node)),
        ]

        if any(d is self for d in self.deps):
            raise ValueError("Cycle detected in DAG")

        child_hashes = tuple(d._hash for d in self.deps)
        raw = (
            self.fn.__qualname__,
            _canonical_args(self),
            child_hashes,
        )
        self._raw = raw
        _hash = hashlib.blake2b(repr(raw).encode(), digest_size=16).hexdigest()
        self._hash = int(_hash, 16)
        self._lock = threading.Lock()

        ancestors: set[Node] = set()
        for d in self.deps:
            anc = getattr(d, "_ancestors", None)
            if anc:
                ancestors.update(anc)
            ancestors.add(d)
        if self in ancestors:
            raise ValueError("Cycle detected in DAG")
        self._ancestors = ancestors

    # --------------------------------------------------------------
    def __getstate__(self):
        return {
            k: getattr(self, k) for k in self.__slots__ if k not in {"flow", "_lock"}
        }

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)
        self.flow = None
        self._lock = threading.Lock()

    def _require_flow(self) -> "Flow":
        if self.flow is None:
            raise RuntimeError("Node has no associated Flow")
        return self.flow

    def __repr__(self):
        return self.signature

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return False
        if self._hash != other._hash:
            return False
        if self._raw != other._raw:
            warnings.warn("Hash collision detected between Nodes", RuntimeWarning)
            return False
        return True

    def __hash__(self) -> int:
        return self._hash

    @property
    def key(self) -> str:
        """Unique identifier combining function name and hash."""
        return f"{self.fn.__name__}_{self._hash:x}"

    def __lt__(self, other: "Node") -> bool:
        return self._hash < other._hash

    @functools.cached_property
    def lines(self) -> List[Tuple[int, str]]:
        """Return script lines for this node without trailing call."""
        with self._lock:
            order = self.order
            lines: List[Tuple[int, str]] = []
            for node in order:
                var_map = {d: d.key for d in node.deps}
                ignore = getattr(node.fn, "_node_ignore", ())
                call = _render_call(
                    node.fn,
                    node.args,
                    node.kwargs,
                    canonical=True,
                    mapping=var_map,
                    ignore=ignore,
                )
                lines.append((node._hash, f"{node.key} = {call}"))
            return lines

    @functools.cached_property
    def order(self) -> List["Node"]:
        return _topo_order(self)

    @functools.cached_property
    def signature(self) -> str:
        return "\n".join(line for _, line in self.lines)

    def get(self):
        return self._require_flow().run(self)

    def delete_cache(self) -> None:
        self._require_flow().engine.cache.delete(self.key)

    def generate(self) -> None:
        """Compute and cache this node without returning the value."""
        self._require_flow().generate(self)


# ----------------------------------------------------------------------
# DAG helpers
# ----------------------------------------------------------------------
def _topo_order(root: Node):
    stack, edges, seen = deque([root]), {}, set()
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        deps = sorted(n.deps)
        edges[n] = deps
        stack.extend(deps)
    return list(TopologicalSorter(edges).static_order())


def _render_call(
    fn: Callable,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    *,
    canonical: bool = False,
    mapping: Dict[Node, str] | None = None,
    ignore: Sequence[str] | None = None,
) -> str:
    """Render a function call with argument names."""

    def render(v: Any) -> str:
        if isinstance(v, Node):
            if mapping:
                return mapping[v]
            return _render_call(
                v.fn,
                v.args,
                v.kwargs,
                canonical=canonical,
                ignore=getattr(v.fn, "_node_ignore", ()),
            )
        return _canonical(v) if canonical else repr(v)

    def key_of(v: Any) -> str:
        if isinstance(v, Node):
            return v.key
        return repr(v)

    key = (
        fn.__qualname__,
        canonical,
        tuple(key_of(a) for a in args),
        tuple(sorted((k, key_of(v)) for k, v in kwargs.items())),
        tuple(sorted((d._hash, v) for d, v in mapping.items())) if mapping else None,
        tuple(sorted(ignore or [])),
    )
    with _ren_lock:
        res = _render_cache.get(key)
    if res is not None:
        return res

    bound = inspect.signature(fn).bind_partial(*args, **kwargs)
    ignore_set = set(ignore or ())
    parts = [
        f"{k}={render(v)}" for k, v in bound.arguments.items() if k not in ignore_set
    ]
    res = f"{fn.__name__}({', '.join(parts)})"

    with _ren_lock:
        _render_cache[key] = res
    return res


def _call_fn(fn: Callable, args: Sequence[Any], kwargs: Mapping[str, Any]) -> Any:
    return fn(*args, **kwargs)


# ----------------------------------------------------------------------
# engine
# ----------------------------------------------------------------------
class Engine:
    def __init__(
        self,
        cache: Cache | None = None,
        *,
        executor: str = "thread",
        workers: int | None = None,
        on_node_start: Callable[[Node], None] | None = None,
        on_node_end: Callable[[Node, float, bool], None] | None = None,
        on_flow_end: Callable[[Node, float, int], None] | None = None,
    ):
        self.cache = cache or ChainCache([MemoryLRU(), DiskJoblib()])
        self.executor = executor
        self.workers = workers or (os.cpu_count() or 4)
        self.on_node_start = on_node_start
        self.on_node_end = on_node_end
        self.on_flow_end = on_flow_end
        self._can_save = hasattr(self.cache, "save_script")

        self._exec_count = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def _resolve(self, v):
        if isinstance(v, Node):
            hit, val = self.cache.get(v.key)
            return val if hit else None
        return v

    def _eval_node(self, n: Node):
        start = time.perf_counter()
        if self.on_node_start is not None:
            self.on_node_start(n)
        hit, val = self.cache.get(n.key)
        if hit:
            dur = time.perf_counter() - start
            if self.on_node_end is not None:
                self.on_node_end(n, dur, True)
            return val

        args = [self._resolve(a) for a in n.args]
        kwargs = {k: self._resolve(v) for k, v in n.kwargs.items()}
        val = n.fn(*args, **kwargs)
        self.cache.put(n.key, val)
        if self._can_save:
            self.cache.save_script(n)
        dur = time.perf_counter() - start
        if self.on_node_end is not None:
            self.on_node_end(n, dur, False)
        with self._lock:
            self._exec_count += 1
        return val

    # ------------------------------------------------------------------
    def run(self, root: Node):
        self._exec_count = 0

        t0 = time.perf_counter()
        hit, val = self.cache.get(root.key)
        if hit:
            if self.on_node_start is not None:
                self.on_node_start(root)
            if self.on_node_end is not None:
                self.on_node_end(root, time.perf_counter() - t0, True)
            if self.on_flow_end is not None:
                self.on_flow_end(root, time.perf_counter() - t0, 0)
            return val

        t0 = time.perf_counter()
        order = root.order

        orig_start = self.on_node_start
        orig_end = self.on_node_end

        def start_cb(n):
            if orig_start is not None:
                orig_start(n)

        def end_cb(n, dur, cached):
            if orig_end is not None:
                orig_end(n, dur, cached)

        self.on_node_start = start_cb
        self.on_node_end = end_cb

        pool_cls = (
            ThreadPoolExecutor if self.executor == "thread" else ProcessPoolExecutor
        )
        ts = TopologicalSorter({n: n.deps for n in order})
        ts.prepare()

        fut_map = {}
        with pool_cls(max_workers=self.workers) as pool:

            def submit(node):
                if self.executor == "thread":
                    fut_map[pool.submit(self._eval_node, node)] = node
                else:
                    start = time.perf_counter()
                    if self.on_node_start:
                        self.on_node_start(node)
                    hit, val = self.cache.get(node.key)
                    if hit:
                        if self.on_node_end:
                            self.on_node_end(node, time.perf_counter() - start, True)
                        ts.done(node)
                        return
                    args = [self._resolve(a) for a in node.args]
                    kwargs = {k: self._resolve(v) for k, v in node.kwargs.items()}
                    fut = pool.submit(_call_fn, node.fn, args, kwargs)
                    fut_map[fut] = (node, start)

            for n in ts.get_ready():
                submit(n)

            while fut_map:
                done, _ = wait(fut_map, return_when=FIRST_COMPLETED)
                for fut in done:
                    info = fut_map.pop(fut)
                    if self.executor == "thread":
                        node = info
                        fut.result()  # re-raise errors immediately
                    else:
                        node, start = info
                        val = fut.result()
                        self.cache.put(node.key, val)
                        if self._can_save:
                            self.cache.save_script(node)
                        dur = time.perf_counter() - start
                        if self.on_node_end is not None:
                            self.on_node_end(node, dur, False)
                        with self._lock:
                            self._exec_count += 1
                    ts.done(node)
                for n in ts.get_ready():
                    submit(n)

        wall = time.perf_counter() - t0
        self.on_node_start = orig_start
        self.on_node_end = orig_end
        if self.on_flow_end is not None:
            self.on_flow_end(root, wall, self._exec_count)
        return self.cache.get(root.key)[1]


# ----------------------------------------------------------------------
# public Flow facade
# ----------------------------------------------------------------------
class Config:
    def __init__(self, mapping: Mapping[str, Dict[str, Any]] | None = None):
        self._m = dict(mapping or {})

    def defaults(self, fn_name: str):
        return self._m.get(fn_name, {})


class Flow:
    def __init__(
        self,
        *,
        config: Config | None = None,
        cache: Cache | None = None,
        executor: str = "thread",
        workers: int | None = None,
        reporter: Optional[Any] = None,
    ):
        self.config = config or Config()
        self.engine = Engine(cache=cache, executor=executor, workers=workers)
        self._registry: WeakValueDictionary[Node, Node] = WeakValueDictionary()
        if reporter is None:
            try:  # defer import to avoid cycle
                from .reporters import RichReporter as _RR
            except Exception:
                self.reporter = None
            else:
                self.reporter = _RR()
        else:
            self.reporter = reporter

    def node(
        self, *, ignore: Sequence[str] | None = None
    ) -> Callable[[Callable[..., Any]], Callable[..., Node]]:
        ignore_set = set(ignore or [])

        def deco(fn: Callable[..., Any]) -> Callable[..., Node]:
            setattr(fn, "_node_ignore", ignore_set)  # type: ignore[attr-defined]
            sig_obj = inspect.signature(fn)
            setattr(fn, "_node_sig", sig_obj)

            @functools.wraps(fn)
            def wrapper(*args, **kwargs) -> Node:
                bound = sig_obj.bind_partial(*args, **kwargs)
                for name, val in self.config.defaults(fn.__name__).items():
                    if name not in bound.arguments:
                        bound.arguments[name] = val
                bound.apply_defaults()

                node = Node(fn, bound.args, bound.kwargs, flow=self)
                cached = self._registry.get(node)
                if cached is not None:
                    return cached
                self._registry[node] = node
                return node

            wrapper.__signature__ = sig_obj  # type: ignore[attr-defined]
            return wrapper

        return deco

    task = node

    def run(self, root: Node, *, reporter=None):
        """Run the DAG rooted at ``root``.

        If ``reporter`` is provided, it should be an object with an
        ``attach(engine, root)`` method returning a context manager that hooks
        into execution callbacks.  When not supplied, :class:`Flow` will use the
        reporter configured during initialization.  By default, this is
        :class:`RichReporter` if ``rich`` is installed.
        """
        if reporter is None:
            reporter = self.reporter
        if reporter is None:
            return self.engine.run(root)
        with reporter.attach(self.engine, root):
            return self.engine.run(root)

    def generate(self, root: Node) -> None:
        """Compute and cache ``root`` without returning the value."""
        self.engine.run(root)

    def clear_caches(self) -> None:
        """Clear global helper caches."""
        with _can_lock:
            _canonical_cache.clear()
        with _ren_lock:
            _render_cache.clear()
