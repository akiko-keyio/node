"""Minimal DAG engine with in-memory and disk caching."""

from __future__ import annotations

import enum
import hashlib
import inspect
import functools
import os
import threading
import time
from collections import defaultdict, deque, OrderedDict
from collections.abc import Mapping, Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from contextlib import nullcontext, suppress
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple
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
class _Sentinel(enum.Enum):
    MISS = enum.auto()


MISS = _Sentinel.MISS

# global caches & locks
_can_lock = threading.Lock()
_ren_lock = threading.Lock()
_sig_lock = threading.Lock()
_canonical_cache: LRUCache[tuple[int, str], str] = LRUCache(maxsize=4096)
_render_cache: LRUCache[tuple, str] = LRUCache(maxsize=2048)


class _Lines:
    __slots__ = ("lines", "__weakref__")

    def __init__(self, lines: List[Tuple[str, str]]):
        self.lines = lines


_signature_cache: WeakValueDictionary[str, _Lines] = WeakValueDictionary()


def _canonical(obj: Any) -> str:
    """Convert an object into a deterministic string representation."""
    if isinstance(obj, Node):
        return obj.signature

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
    bound = inspect.signature(node.fn).bind_partial(*node.args, **node.kwargs)
    ignore = set(getattr(node.fn, "_node_ignore", ()))
    parts = []
    for k, v in bound.arguments.items():
        if k in ignore:
            continue
        if isinstance(v, Node):
            parts.append((k, v._hash))
        else:
            parts.append((k, _canonical(v)))
    return tuple(parts)


def _merge_lines(nodes: Sequence["Node"]) -> List[Tuple[str, str]]:
    merged: OrderedDict[str, str] = OrderedDict()
    for n in sorted(nodes, key=lambda x: x._hash):
        for h, line in n.lines():
            merged.setdefault(h, line)
    return [(h, line) for h, line in merged.items()]


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
        return False, MISS

    def put(self, key: str, value: Any):
        with self._lock:
            self._lru[key] = value

    def delete(self, key: str) -> None:
        with self._lock:
            self._lru.pop(key, None)


class DiskJoblib(Cache):
    """Filesystem cache using joblib pickles.

    Results are stored under ``<func>/<expr>.pkl`` whenever possible.  If the
    expression cannot be used as a file name, the fallback ``<hash>.pkl`` is
    used and ``<hash>.py`` contains ``repr(node)`` for inspection.

    Results are stored as ``<hash>.pkl``.  A human readable ``<hash>.py`` file
    containing ``repr(node)`` is also written for inspection.
    """

    def __init__(self, root: str | Path = ".cache", lock: bool = True):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = lock

    def _subdir(self, key: str) -> Path:
        """Return the directory corresponding to the node's function name."""
        lines = key.strip().splitlines()[-1]
        fn_name = lines.split("(", 1)[0]
        sub = self.root / fn_name
        sub.mkdir(parents=True, exist_ok=True)
        return sub

    def _expr_path(self, key: str, ext: str = ".pkl") -> Path:
        return self._subdir(key) / (key + ext)

    def _hash_path(self, key: str, ext: str = ".pkl") -> Path:
        md = hashlib.md5(key.encode()).hexdigest()
        return self._subdir(key) / (md + ext)

    def get(self, key: str):
        for p in (self._expr_path(key), self._hash_path(key)):
            if p.exists():
                return True, joblib.load(p)
        return False, MISS

    def put(self, key: str, value: Any):
        for path_fn in (self._expr_path, self._hash_path):
            p = path_fn(key)
            lock_path = str(p) + ".lock"
            ctx = FileLock(lock_path) if self.lock else nullcontext()
            try:
                with ctx:
                    joblib.dump(value, p)
                return
            except OSError:
                continue

    def delete(self, key: str) -> None:
        hash_p = self._hash_path(key)
        for p in (self._expr_path(key), hash_p, hash_p.with_suffix(".py")):
            if p.exists():
                with suppress(OSError):
                    p.unlink()

    def save_script(self, node: "Node"):
        if self._expr_path(node.signature).exists():
            return
        p = self._hash_path(node.signature, ".py")
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
        return False, MISS

    def put(self, key: str, value: Any):
        for c in self.caches:
            c.put(key, value)

    def delete(self, key: str) -> None:
        for c in self.caches:
            if hasattr(c, "delete"):
                c.delete(key)

    def save_script(self, node: "Node"):
        for c in self.caches:
            if hasattr(c, "save_script"):
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
        "_lines",
        "__signature",
        "_lock",
        "__weakref__",
    )

    _hash: str
    _lines: List[Tuple[str, str]] | None
    __signature: str | None
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
        self._detect_cycle()

        child_hashes = tuple(d._hash for d in self.deps)
        code_hash = hashlib.blake2b(self.fn.__code__.co_code, digest_size=8).hexdigest()
        raw = (
            self.fn.__qualname__,
            code_hash,
            _canonical_args(self),
            child_hashes,
        )
        self._hash = hashlib.blake2b(repr(raw).encode(), digest_size=16).hexdigest()
        self._lines: List[Tuple[str, str]] | None = None
        self.__signature: str | None = None
        self._lock = threading.Lock()

    def _require_flow(self) -> "Flow":
        if self.flow is None:
            raise RuntimeError("Node has no associated Flow")
        return self.flow

    def _detect_cycle(self):
        try:
            _topo_order(self)
        except CycleError as e:
            raise ValueError("Cycle detected in DAG") from e

    def __repr__(self):
        return self.signature

    @property
    def var(self) -> str:
        return f"h_{self._hash[:6]}"

    def lines(self) -> List[Tuple[str, str]]:
        """Return script lines for this node without trailing call."""
        with self._lock:
            cached = self._lines
            if cached is not None:
                return cached
            holder = _signature_cache.get(self._hash)
            if holder is not None:
                self._lines = holder.lines
                return holder.lines

        return self._compute_lines()

    def _collect_lines(self) -> OrderedDict[str, str]:
        merged: OrderedDict[str, str] = OrderedDict()
        for dep in self.deps:
            for h, line in dep.lines():
                merged.setdefault(h, line)
        return merged

    def _compute_lines(self) -> List[Tuple[str, str]]:
        merged = self._collect_lines()
        var_map = {d: d.var for d in self.deps}
        ignore = getattr(self.fn, "_node_ignore", ())
        call = _render_call(
            self.fn,
            self.args,
            self.kwargs,
            canonical=True,
            mapping=var_map,
            ignore=ignore,
        )
        merged[self._hash] = f"{self.var} = {call}"
        lines = [(h, line) for h, line in merged.items()]
        holder = _Lines(lines)
        with _sig_lock:
            _signature_cache[self._hash] = holder
        with self._lock:
            self._lines = lines
        return lines

    def _compute_signature(self) -> str:
        with self._lock:
            if self.__signature is not None:
                return self.__signature

        if _is_linear_chain(self):
            var_map = {d: d.var for d in self.deps}
            ignore = getattr(self.fn, "_node_ignore", ())
            result = _render_call(
                self.fn,
                self.args,
                self.kwargs,
                canonical=True,
                mapping=var_map,
                ignore=ignore,
            )
        else:
            result = self._non_linear_signature()

        with self._lock:
            self.__signature = result
        return result

    def _non_linear_signature(self) -> str:
        self.lines()  # ensure populated
        var_map = {d: d.var for d in self.deps}
        ignore = getattr(self.fn, "_node_ignore", ())
        call = _render_call(
            self.fn,
            self.args,
            self.kwargs,
            canonical=True,
            mapping=var_map,
            ignore=ignore,
        )
        lines = [line for _, line in self.lines()[:-1]]
        lines.append(call)
        return "\n".join(lines)

    @property
    def signature(self) -> str:
        return self._compute_signature()

    def get(self):
        return self._require_flow().run(self)

    def delete_cache(self) -> None:
        flow = self._require_flow()
        if hasattr(flow.engine.cache, "delete"):
            flow.engine.cache.delete(self.signature)

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
        deps = sorted(n.deps, key=lambda x: getattr(x, "_hash", ""))
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

    key = (
        fn.__qualname__,
        canonical,
        tuple(repr(a) for a in args),
        tuple(sorted((k, repr(v)) for k, v in kwargs.items())),
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


def _is_linear_chain(root: Node) -> bool:
    """Return ``True`` if the graph rooted at ``root`` has no diamond dependencies."""

    indeg: Dict[str, int] = defaultdict(int)
    seen: set[Node] = set()
    stack = [root]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        for d in n.deps:  # ``deps`` already contains only ``Node`` objects
            indeg[d._hash] += 1
            if indeg[d._hash] > 1:
                return False
            stack.append(d)
    return True


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
        log: bool = True,
        on_node_start: Callable[[Node], None] | None = None,
        on_node_end: Callable[[Node, float, bool], None] | None = None,
        on_flow_end: Callable[[Node, float, int], None] | None = None,
    ):
        self.cache = cache or ChainCache([MemoryLRU(), DiskJoblib()])
        self.executor = executor
        self.workers = workers or (os.cpu_count() or 4)
        self.log = log
        self.on_node_start = on_node_start
        self.on_node_end = on_node_end
        self.on_flow_end = on_flow_end
        self._can_save = hasattr(self.cache, "save_script")

        self._exec_count = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def _resolve(self, v):
        if isinstance(v, Node):
            hit, val = self.cache.get(v.signature)
            return val if hit else None
        return v

    def _eval_node(self, n: Node):
        start = time.perf_counter()
        if self.on_node_start:
            self.on_node_start(n)
        hit, val = self.cache.get(n.signature)
        if hit:
            dur = time.perf_counter() - start
            if self.on_node_end:
                self.on_node_end(n, dur, True)
            return val

        args = [self._resolve(a) for a in n.args]
        kwargs = {k: self._resolve(v) for k, v in n.kwargs.items()}
        val = n.fn(*args, **kwargs)
        self.cache.put(n.signature, val)
        if self._can_save:
            self.cache.save_script(n)
        dur = time.perf_counter() - start
        if self.on_node_end:
            self.on_node_end(n, dur, False)
        with self._lock:
            self._exec_count += 1
        return val

    # ------------------------------------------------------------------
    def run(self, root: Node):
        self._exec_count = 0

        t0 = time.perf_counter()
        hit, val = self.cache.get(root.signature)
        if hit:
            if self.on_node_start:
                self.on_node_start(root)
            if self.on_node_end:
                self.on_node_end(root, time.perf_counter() - t0, True)
            if self.on_flow_end:
                self.on_flow_end(root, time.perf_counter() - t0, 0)
            return val

        t0 = time.perf_counter()
        order = _topo_order(root)

        orig_start = self.on_node_start
        orig_end = self.on_node_end

        def start_cb(n):
            if orig_start:
                orig_start(n)

        def end_cb(n, dur, cached):
            if orig_end:
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
                fut_map[pool.submit(self._eval_node, node)] = node

            for n in ts.get_ready():
                submit(n)

            while fut_map:
                done, _ = wait(fut_map, return_when=FIRST_COMPLETED)
                for fut in done:
                    node = fut_map.pop(fut)
                    fut.result()  # re-raise errors immediately
                    ts.done(node)
                for n in ts.get_ready():
                    submit(n)

        wall = time.perf_counter() - t0
        self.on_node_start = orig_start
        self.on_node_end = orig_end
        if self.on_flow_end:
            self.on_flow_end(root, wall, self._exec_count)
        return self.cache.get(root.signature)[1]


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
        log: bool = True,
        reporter=_Sentinel.MISS,
    ):
        self.config = config or Config()
        self.engine = Engine(cache=cache, executor=executor, workers=workers, log=log)
        self._registry: WeakValueDictionary[str, Node] = WeakValueDictionary()
        self.log = log
        if reporter is _Sentinel.MISS:
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

            @functools.wraps(fn)
            def wrapper(*args, **kwargs) -> Node:
                bound = sig_obj.bind_partial(*args, **kwargs)
                for name, val in self.config.defaults(fn.__name__).items():
                    if name not in bound.arguments:
                        bound.arguments[name] = val
                bound.apply_defaults()

                node = Node(fn, bound.args, bound.kwargs, flow=self)
                cached = self._registry.get(node._hash)
                if cached is not None:
                    return cached
                self._registry[node._hash] = node
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
        with _sig_lock:
            _signature_cache.clear()
