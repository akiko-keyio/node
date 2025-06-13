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
import os
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from contextlib import contextmanager
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


@contextmanager
def _nullcontext():
    yield


# ----------------------------------------------------------------------
# cache abstractions
# ----------------------------------------------------------------------
class Cache:
    def get(self, key: str) -> Tuple[bool, Any]: ...
    def put(self, key: str, value: Any) -> None: ...


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


class DiskJoblib(Cache):
    """Filesystem cache using joblib pickles.

    When ``pretty`` is True, cache file names include a sanitized snippet of the
    cache key before the MD5 hash so they are somewhat readable.
    """

    def __init__(self, root: str | Path = ".cache", lock: bool = True, pretty: bool = False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = lock
        self.pretty = pretty

    def _sanitize(self, text: str) -> str:
        safe = [c if c.isalnum() or c in "-_." else "_" for c in text]
        return "".join(safe)

    def _path(self, key: str) -> Path:
        md = hashlib.md5(key.encode()).hexdigest()
        if self.pretty:
            prefix = self._sanitize(key)[:32]
            name = f"{prefix}-{md}.pkl"
        else:
            name = md + ".pkl"
        sub = self.root / md[:2]
        sub.mkdir(exist_ok=True)
        return sub / name

    def get(self, key: str):
        p = self._path(key)
        if p.exists():
            return True, joblib.load(p)
        return False, MISS

    def put(self, key: str, value: Any):
        p = self._path(key)
        lock_path = str(p) + ".lock"
        ctx = FileLock(lock_path) if self.lock else _nullcontext()
        with ctx:
            joblib.dump(value, p)


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


# ----------------------------------------------------------------------
# DAG nodes
# ----------------------------------------------------------------------
class Node:
    __slots__ = ("fn", "args", "kwargs", "deps", "signature", "__weakref__")

    def __init__(self, fn, args: Tuple = (), kwargs: Dict | None = None):
        self.fn = fn
        self.args = tuple(args)
        self.kwargs = kwargs or {}

        self.deps: List[Node] = [
            *(a for a in self.args if isinstance(a, Node)),
            *(v for v in self.kwargs.values() if isinstance(v, Node)),
        ]
        self._detect_cycle(set())

        pieces: List[str] = []
        params = inspect.signature(fn).parameters
        for name, arg in zip(params, self.args):
            pieces.append(f"{name}={_canonical(arg)}")
        for name in sorted(self.kwargs):
            pieces.append(f"{name}={_canonical(self.kwargs[name])}")
        self.signature = f"{fn.__name__}({', '.join(pieces)})"

    def _detect_cycle(self, anc: set):
        if self in anc:
            raise ValueError("Cycle detected in DAG")
        anc.add(self)
        for d in self.deps:
            d._detect_cycle(anc)
        anc.remove(self)

    def __repr__(self):
        script, _ = _build_script(self)
        return script


# ----------------------------------------------------------------------
# DAG helpers
# ----------------------------------------------------------------------
def _topo_order(root: Node):
    out, seen = [], set()

    def dfs(n: Node):
        if n in seen:
            return
        seen.add(n)
        for d in sorted(n.deps, key=lambda x: x.signature):
            dfs(d)
        out.append(n)

    dfs(root)
    return out


def _build_script(root: Node):
    order = _topo_order(root)
    sig2var: Dict[str, str] = {}
    mapping: Dict[Node, str] = {}
    lines: List[str] = []

    def rend(x):
        return mapping[x] if isinstance(x, Node) else repr(x)

    for n in order:
        if n.signature in sig2var:
            mapping[n] = sig2var[n.signature]
            continue
        var = f"n{len(sig2var)}"
        sig2var[n.signature] = var
        mapping[n] = var
        args_s = ", ".join(rend(a) for a in n.args)
        kw_s = ", ".join(f"{k}={rend(v)}" for k, v in n.kwargs.items())
        call = ", ".join(filter(None, (args_s, kw_s)))
        lines.append(f"{var} = {n.fn.__name__}({call})")

    lines.append(mapping[root])
    return "\n".join(lines), mapping


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

        indeg = {n: len(n.deps) for n in order}
        succ = {n: [] for n in order}
        for n in order:
            for d in n.deps:
                succ[d].append(n)

        ready = sorted((n for n, d in indeg.items() if d == 0), key=lambda x: x.signature)
        fut_map = {}
        pool_cls = ThreadPoolExecutor if self.executor == "thread" else ProcessPoolExecutor

        with pool_cls(max_workers=self.workers) as pool:
            def submit(node):
                fut_map[pool.submit(self._eval_node, node)] = node

            for n in ready:
                submit(n)

            while fut_map:
                done, _ = wait(fut_map, return_when=FIRST_COMPLETED)
                for fut in done:
                    node = fut_map.pop(fut)
                    fut.result()          # re-raise errors immediately
                    for nxt in succ[node]:
                        indeg[nxt] -= 1
                        if indeg[nxt] == 0:
                            submit(nxt)

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

    def task(self):
        def deco(fn):
            sig_obj = inspect.signature(fn)

            def wrapper(*args, **kwargs):
                merged = {**self.config.defaults(fn.__name__), **kwargs}
                bound = sig_obj.bind_partial(*args, **merged)
                bound.apply_defaults()

                node = Node(fn, bound.args, bound.kwargs)
                cached = self._registry.get(node.signature)
                if cached is not None:
                    return cached
                self._registry[node.signature] = node
                return node

            return wrapper

        return deco

    def run(self, root: Node):
        return self.engine.run(root)


# ----------------------------------------------------------------------
# simple example
# ----------------------------------------------------------------------
if __name__ == "__main__":
    flow = Flow()

    @flow.task()
    def add(x, y):
        return x + y

    @flow.task()
    def square(z):
        return z * z

    out = flow.run(square(add(2, 3)))
    print("result =", out)  # 25