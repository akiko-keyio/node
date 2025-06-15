"""Minimal DAG engine with in-memory and disk caching."""

from __future__ import annotations

import enum
import hashlib
import inspect
import functools
import os
import threading
import time
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from contextlib import nullcontext, suppress
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Plan:
    order: tuple["Node", ...]
    edges: dict["Node", tuple["Node", ...]]
    mapping: dict["Node", str]
    lines: tuple[str, ...]
    shown: tuple["Node", ...]


_PLAN_CACHE: LRUCache[int, Plan] = LRUCache(maxsize=128)
_PLAN_LOCK = threading.Lock()


def _build_plan(root: "Node") -> Plan:
    order, edges = _topo_order(root, return_edges=True)
    _, mapping, _calls, _dups, lines, shown = _plan_dag(root, order=order)
    is_linear = _is_linear_chain(edges)
    if is_linear:
        ignore = getattr(root.fn, "_node_ignore", ())
        root_sig = _render_call(
            root.fn, root.args, root.kwargs, canonical=True, ignore=ignore
        )
        mapping[root] = root_sig
        lines = [mapping[n] for n in order]
        shown = order
    else:
        root_sig = "\n".join(lines)
        mapping[root] = root_sig
    if not hasattr(root, "_signature"):
        root.signature = root_sig
    edge_map = {n: tuple(edges[n]) for n in order}
    return Plan(tuple(order), edge_map, mapping, tuple(lines), tuple(shown))


def get_plan(root: "Node") -> Plan:
    root_id = id(root)
    plan = _PLAN_CACHE.get(root_id)
    if plan is not None:
        return plan
    with _PLAN_LOCK:
        plan = _PLAN_CACHE.get(root_id)
        if plan is None:
            plan = _build_plan(root)
            _PLAN_CACHE[root_id] = plan
        return plan


def _canonical(obj: Any) -> str:
    """Convert an object into a deterministic string representation."""
    if isinstance(obj, Node):
        return obj.signature
    if isinstance(obj, dict):
        inner = ", ".join(f"{repr(k)}: {_canonical(v)}" for k, v in sorted(obj.items()))
        return "{" + inner + "}"
    if isinstance(obj, (list, tuple)):
        inner = ", ".join(_canonical(v) for v in obj)
        return "[" + inner + "]" if isinstance(obj, list) else "(" + inner + ")"
    if isinstance(obj, set):
        inner = ", ".join(_canonical(v) for v in sorted(obj))
        return "{" + inner + "}"
    return repr(obj)


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

    def _dir_for(self, key: str) -> Path:
        """Return the directory for ``key`` and ensure it exists."""
        lines = key.strip().splitlines()[-1]
        fn_name = lines.split("(", 1)[0]
        sub = self.root / fn_name
        sub.mkdir(parents=True, exist_ok=True)
        return sub

    def _path(self, key: str, *, hashed: bool = False, ext: str = ".pkl") -> Path:
        name = hashlib.md5(key.encode()).hexdigest() if hashed else key
        return self._dir_for(key) / (name + ext)

    def _expr_path(self, key: str, ext: str = ".pkl") -> Path:
        return self._path(key, hashed=False, ext=ext)

    def _hash_path(self, key: str, ext: str = ".pkl") -> Path:
        return self._path(key, hashed=True, ext=ext)

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
        "_signature",
        "flow",
        "__weakref__",
    )

    @property
    def signature(self) -> str:
        return self._signature

    @signature.setter
    def signature(self, val: str) -> None:
        if hasattr(self, "_signature"):
            raise AttributeError("signature is immutable")
        self._signature = val

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
        if self in self.deps:
            raise ValueError("Node cannot depend on itself")
        sig = get_plan(self).mapping[self]
        if not hasattr(self, "_signature"):
            self.signature = sig

    def _require_flow(self) -> "Flow":
        if self.flow is None:
            raise RuntimeError("Node has no associated Flow")
        return self.flow

    def __repr__(self):
        return self.signature

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
def _topo_order(root: Node, *, return_edges: bool = False):
    stack, edges, seen = deque([root]), {}, set()
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        deps = sorted(n.deps, key=lambda x: getattr(x, "signature", ""))
        edges[n] = deps
        stack.extend(deps)
    try:
        order = list(TopologicalSorter(edges).static_order())
    except CycleError as e:  # pragma: no cover - rare
        raise ValueError("Cycle detected in DAG") from e
    return (order, edges) if return_edges else order


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

    bound = inspect.signature(fn).bind_partial(*args, **kwargs)
    ignore_set = set(ignore or ())
    parts = [
        f"{k}={render(v)}" for k, v in bound.arguments.items() if k not in ignore_set
    ]
    return f"{fn.__name__}({', '.join(parts)})"


def _node_key(n: Node) -> str:
    ignore = getattr(n.fn, "_node_ignore", ())
    return getattr(n, "signature", None) or _render_call(
        n.fn, n.args, n.kwargs, canonical=True, ignore=ignore
    )


def _plan_dag(
    root: Node,
    order: List[Node] | None = None,
) -> tuple[
    list[Node],
    dict[Node, str],
    dict[Node, str],
    set[Node],
    list[str],
    list[Node],
]:
    order = order or _topo_order(root)
    sig2var: Dict[str, str] = {}
    mapping: Dict[Node, str] = {}
    calls: Dict[Node, str] = {}
    dups: set[Node] = set()
    lines: List[str] = []
    shown: List[Node] = []
    for n in order:
        key = _node_key(n)
        if key in sig2var:
            mapping[n] = sig2var[key]
            dups.add(n)
        else:
            mapping[n] = key if n is root else f"n{len(sig2var)}"
            if n is not root:
                sig2var[key] = mapping[n]
        call = _render_call(
            n.fn,
            n.args,
            n.kwargs,
            canonical=True,
            mapping=mapping,
            ignore=getattr(n.fn, "_node_ignore", ()),
        )
        calls[n] = call
        if n in dups and n is not root:
            continue
        lines.append(call if n is root else f"{mapping[n]} = {call}")
        shown.append(n)
    return order, mapping, calls, dups, lines, shown


def _build_script(root: Node, *, order: List[Node] | None = None):
    _, mapping, _, _, lines, _ = _plan_dag(root, order=order)
    return "\n".join(lines), mapping


def _is_linear_chain(edges: Dict[Node, List[Node]]) -> bool:
    """Return ``True`` if no node in ``edges`` has multiple parents."""

    indeg: Dict[str, int] = defaultdict(int)
    for deps in edges.values():
        for d in deps:
            key = _node_key(d)
            indeg[key] += 1
            if indeg[key] > 1:
                return False
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
        self._val_lock = threading.Lock()
        if executor == "thread":
            self._values: Dict[Node, Any] | None = {}
        else:
            self._values = None

    # ------------------------------------------------------------------
    def _resolve(self, v, *, check_cache: bool = True):
        """Return cached value for ``v`` or :data:`MISS`."""
        if not isinstance(v, Node):
            return v
        if check_cache and self._values is not None:
            with self._val_lock:
                if v in self._values:
                    return self._values[v]
            hit, val = self.cache.get(v.signature)
            if hit:
                with self._val_lock:
                    self._values[v] = val
                return val
        elif check_cache:
            hit, val = self.cache.get(v.signature)
            if hit:
                return val
        return MISS

    def _eval_node(self, n: Node, *, skip_cache: bool = False):
        start = time.perf_counter()
        if self.on_node_start:
            self.on_node_start(n)
        cached = self._resolve(n, check_cache=not skip_cache)
        if cached is not MISS:
            return self._finish_node(n, start, cached, True)

        args = [self._resolve(a) for a in n.args]
        kwargs = {k: self._resolve(v) for k, v in n.kwargs.items()}
        val = n.fn(*args, **kwargs)
        self.cache.put(n.signature, val)
        if self._can_save:
            self.cache.save_script(n)
        if self._values is not None:
            with self._val_lock:
                self._values[n] = val
        return self._finish_node(n, start, val, False)

    def _finish_node(self, n: Node, start: float, val: Any, cached: bool):
        dur = time.perf_counter() - start
        if self.on_node_end:
            self.on_node_end(n, dur, cached)
        with self._lock:
            if not cached:
                self._exec_count += 1
        return val

    # ------------------------------------------------------------------
    def run(self, root: Node):
        self._exec_count = 0
        if self._values is not None:
            with self._val_lock:
                self._values = {}

        t0 = time.perf_counter()
        hit, val = self.cache.get(root.signature)
        if hit:
            self._report_cached(root, t0)
            return val

        plan = get_plan(root)
        edges = {n: list(d) for n, d in plan.edges.items()}
        orig = (self.on_node_start, self.on_node_end)
        self.on_node_start, self.on_node_end = self._wrap_callbacks(orig)
        dep_count, dependents = self._build_schedule(edges)
        root_val = self._run_pool(root, dep_count, dependents)
        wall = time.perf_counter() - t0
        self.on_node_start, self.on_node_end = orig
        if self.on_flow_end:
            self.on_flow_end(root, wall, self._exec_count)
        return root_val

    # ------------------------------------------------------------------
    def _report_cached(self, root: Node, start: float) -> None:
        if self.on_node_start:
            self.on_node_start(root)
        dur = time.perf_counter() - start
        if self.on_node_end:
            self.on_node_end(root, dur, True)
        if self.on_flow_end:
            self.on_flow_end(root, dur, 0)

    def _wrap_callbacks(
        self,
        orig: tuple[
            Callable[[Node], None] | None, Callable[[Node, float, bool], None] | None
        ],
    ) -> tuple[
        Callable[[Node], None] | None, Callable[[Node, float, bool], None] | None
    ]:
        orig_start, orig_end = orig

        def start_cb(n: Node):
            if orig_start:
                orig_start(n)

        def end_cb(n: Node, dur: float, cached: bool):
            if orig_end:
                orig_end(n, dur, cached)

        return start_cb, end_cb

    def _build_schedule(self, edges: Dict[Node, List[Node]]):
        dep_count = {n: len(d) for n, d in edges.items()}
        dependents: Dict[Node, List[Node]] = defaultdict(list)
        for n, deps in edges.items():
            for d in deps:
                dependents[d].append(n)
        return dep_count, dependents

    def _submit(self, pool, node: Node, fut_map: Dict[Any, Node], root: Node) -> None:
        fut = pool.submit(self._eval_node, node, skip_cache=(node is root))
        fut_map[fut] = node

    def _schedule_initial(
        self,
        pool,
        dep_count: Dict[Node, int],
        fut_map: Dict[Any, Node],
        root: Node,
    ) -> None:
        for n, cnt in dep_count.items():
            if cnt == 0:
                self._submit(pool, n, fut_map, root)

    def _process_done(
        self,
        done,
        dep_count: Dict[Node, int],
        dependents: Dict[Node, List[Node]],
        fut_map: Dict[Any, Node],
        pool,
        root: Node,
        root_val: List[Any],
    ) -> None:
        for fut in done:
            node = fut_map.pop(fut)
            val = fut.result()
            if node is root:
                root_val[0] = val
            for nxt in dependents[node]:
                dep_count[nxt] -= 1
                if dep_count[nxt] == 0:
                    self._submit(pool, nxt, fut_map, root)

    def _run_pool(
        self,
        root: Node,
        dep_count: Dict[Node, int],
        dependents: Dict[Node, List[Node]],
    ) -> Any:
        pool_cls = (
            ThreadPoolExecutor if self.executor == "thread" else ProcessPoolExecutor
        )
        fut_map: Dict[Any, Node] = {}
        root_val: List[Any] = [None]
        with pool_cls(max_workers=self.workers) as pool:
            self._schedule_initial(pool, dep_count, fut_map, root)
            while fut_map:
                done, _ = wait(fut_map, return_when=FIRST_COMPLETED)
                self._process_done(
                    done,
                    dep_count,
                    dependents,
                    fut_map,
                    pool,
                    root,
                    root_val,
                )
        return root_val[0]


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
                cached = self._registry.get(node.signature)
                if cached is not None:
                    return cached
                self._registry[node.signature] = node
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
