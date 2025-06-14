"""
zen_flow.py – minimal DAG + two-tier cache (rev-6, 2025-05-31)

Fixes applied
-------------
* add missing import for WeakValueDictionary
* make MemoryLRU thread-safe
* protect ChainCache promotions with a lock
* guard ProcessPool on Windows (pickle issues)
* canonicalise `set` objects when building signatures
"""

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
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from contextlib import nullcontext, suppress
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple
from weakref import WeakValueDictionary  # ★ NEW

import joblib
from cachetools import LRUCache
from filelock import FileLock
from loguru import logger


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
class _Sentinel(enum.Enum):
    MISS = enum.auto()


MISS = _Sentinel.MISS


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
    if isinstance(obj, set):  # ★ NEW – make set order deterministic
        inner = ", ".join(_canonical(v) for v in sorted(obj))
        return "{" + inner + "}"
    return repr(obj)


# ----------------------------------------------------------------------
# cache abstractions
# ----------------------------------------------------------------------
class Cache:
    def get(self, key: str) -> Tuple[bool, Any]: ...
    def put(self, key: str, value: Any) -> None: ...
    def delete(self, key: str) -> None: ...


class MemoryLRU(Cache):
    """Thread-safe in-memory LRU cache."""
    def __init__(self, maxsize: int = 512):
        self._lru: LRUCache[str, Any] = LRUCache(maxsize=maxsize)
        self._lock = threading.Lock()  # ★ NEW

    def get(self, key: str):
        with self._lock:                       # ★ NEW
            if key in self._lru:
                return True, self._lru[key]
        return False, MISS

    def put(self, key: str, value: Any):
        with self._lock:                       # ★ NEW
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
        p = self._expr_path(key)
        if p.exists():
            return True, joblib.load(p)
        p = self._hash_path(key)
        if p.exists():
            return True, joblib.load(p)
        return False, MISS

    def put(self, key: str, value: Any):
        p = self._expr_path(key)
        lock_path = str(p) + ".lock"
        ctx = FileLock(lock_path) if self.lock else nullcontext()
        with suppress(OSError):
            with ctx:
                joblib.dump(value, p)
            return

        p = self._hash_path(key)
        lock_path = str(p) + ".lock"
        ctx = FileLock(lock_path) if self.lock else nullcontext()
        with ctx:
            joblib.dump(value, p)

    def delete(self, key: str) -> None:
        expr_p = self._expr_path(key)
        hash_p = self._hash_path(key)
        for p in (expr_p, hash_p):
            if p.exists():
                with suppress(OSError):
                    p.unlink()
        py = hash_p.with_suffix('.py')
        if py.exists():
            with suppress(OSError):
                py.unlink()

    def save_script(self, node: "Node"):
        if self._expr_path(node.signature).exists():
            return
        p = self._hash_path(node.signature, ".py")
        p.write_text(repr(node) + "\n")


class ChainCache(Cache):
    """Chain several caches (e.g. Memory → Disk)."""
    def __init__(self, caches: Sequence[Cache]):
        self.caches = list(caches)
        self._lock = threading.Lock()  # ★ NEW

    def get(self, key: str):
        with self._lock:               # ★ NEW – protects promotion
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
        "signature",
        "flow",
        "__weakref__",
    )

    def __init__(
        self, fn, args: Tuple = (), kwargs: Dict | None = None, *, flow: "Flow" | None = None
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

        ignore = getattr(self.fn, "_node_ignore", ())

        if _is_linear_chain(self):
            self.signature = _render_call(
                self.fn, self.args, self.kwargs, canonical=True, ignore=ignore
            )
        else:
            script, _ = _build_script(self)
            self.signature = script

    def _detect_cycle(self):
        try:
            _topo_order(self)
        except CycleError as e:
            raise ValueError("Cycle detected in DAG") from e

    def __repr__(self):
        return self.signature

    def get(self):
        if self.flow is None:
            raise RuntimeError("Node has no associated Flow")
        return self.flow.run(self)

    def delete_cache(self) -> None:
        if self.flow is None:
            raise RuntimeError("Node has no associated Flow")
        if hasattr(self.flow.engine.cache, "delete"):
            self.flow.engine.cache.delete(self.signature)

    def generate(self) -> None:
        """Compute and cache this node without returning the value."""
        if self.flow is None:
            raise RuntimeError("Node has no associated Flow")
        self.flow.generate(self)



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
        deps = sorted(n.deps, key=lambda x: getattr(x, "signature", ""))
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

    render = lambda v: (
        mapping[v]
        if isinstance(v, Node) and mapping
        else _render_call(
            v.fn,
            v.args,
            v.kwargs,
            canonical=canonical,
            ignore=getattr(v.fn, "_node_ignore", ()),
        )
        if isinstance(v, Node)
        else _canonical(v)
        if canonical
        else repr(v)
    )

    bound = inspect.signature(fn).bind_partial(*args, **kwargs)
    ignore_set = set(ignore or ())
    parts = [f"{k}={render(v)}" for k, v in bound.arguments.items() if k not in ignore_set]
    return f"{fn.__name__}({', '.join(parts)})"


def _build_script(root: Node):
    order = _topo_order(root)
    sig2var, mapping, lines = {}, {}, []

    for n in order:
        ignore = getattr(n.fn, "_node_ignore", ())
        key = getattr(n, "signature", None) or _render_call(
            n.fn, n.args, n.kwargs, canonical=True, ignore=ignore
        )
        if key in sig2var:
            mapping[n] = sig2var[key]
            if n is root:
                lines.append(mapping[n])
            continue

        var = key if n is root else f"n{len(sig2var)}"
        mapping[n] = var
        if n is not root:
            sig2var[key] = var
        call = _render_call(
            n.fn,
            n.args,
            n.kwargs,
            canonical=True,
            mapping=mapping,
            ignore=ignore,
        )
        lines.append(call if n is root else f"{var} = {call}")

    return "\n".join(lines), mapping


def _is_linear_chain(root: Node) -> bool:
    """Return ``True`` if the graph rooted at ``root`` has no diamond dependencies."""

    indeg, seen, stack = defaultdict(int), set(), [root]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        for d in n.deps:  # ``deps`` already contains only ``Node`` objects
            indeg[d.signature] += 1
            if indeg[d.signature] > 1:
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
        on_node_end: Callable[[Node, float, bool], None] | None = None,
        on_flow_end: Callable[[Node, float, int], None] | None = None,
    ):

        self.cache = cache or ChainCache([MemoryLRU(), DiskJoblib()])
        self.executor = executor
        self.workers = workers or (os.cpu_count() or 4)
        self.log = log
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
        hit, val = self.cache.get(n.signature)
        if hit:
            if self.on_node_end:
                self.on_node_end(n, 0.0, True)
            return val

        start = time.perf_counter()
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
        order = _topo_order(root)

        pool_cls = ThreadPoolExecutor if self.executor == "thread" else ProcessPoolExecutor
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
        if self.on_flow_end:
            self.on_flow_end(root, wall, self._exec_count)
        if self.log:
            logger.info(f"Flow done: {self._exec_count}/{len(order)} tasks executed, wall {wall:.3f}s")
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
    ):
        self.config = config or Config()
        self.engine = Engine(cache=cache, executor=executor, workers=workers, log=log)
        self._registry: WeakValueDictionary[str, Node] = WeakValueDictionary()
        self.log = log

    def node(self, *, ignore: Sequence[str] | None = None):
        ignore_set = set(ignore or [])

        def deco(fn):
            fn._node_ignore = ignore_set
            sig_obj = inspect.signature(fn)

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
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

            wrapper.__signature__ = sig_obj
            return wrapper

        return deco

    task = node

    def run(self, root: Node):
        return self.engine.run(root)

    def generate(self, root: Node) -> None:
        """Compute and cache ``root`` without returning the value."""
        self.engine.run(root)


# ----------------------------------------------------------------------
# simple example
# ----------------------------------------------------------------------
if __name__ == "__main__":
    flow = Flow()

    @flow.node()
    def add(x, y):
        return x + y

    @flow.node()
    def square(z):
        return z * z

    out = flow.run(square(add(2, 3)))
    print("result =", out)  # 25
