"""Minimal DAG engine with in-memory and disk caching."""

from __future__ import annotations

import hashlib
import inspect
import functools
import os
import pickle
import threading
import time
import warnings
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    ThreadPoolExecutor,
    wait,
)
from loky import ProcessPoolExecutor
from contextlib import nullcontext, suppress
from graphlib import TopologicalSorter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, cast
from omegaconf import DictConfig, OmegaConf
from pydantic import validate_arguments
from weakref import WeakValueDictionary

import joblib  # type: ignore[import]
from cachetools import LRUCache  # type: ignore[import]
from filelock import FileLock  # type: ignore[import]
from .logger import logger

__all__ = [
    "Cache",
    "MemoryLRU",
    "DiskJoblib",
    "ChainCache",
    "Node",
    "Config",
    "Flow",
    "gather",
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
    ignore = set(getattr(node.fn, "_node_ignore", ()))
    parts = []
    for k, v in node.bound_args.items():
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
        if not p.exists():
            return False, None

        if p.stat().st_size <= self.small_file:
            with p.open("rb") as fh:
                return True, pickle.load(fh)

        return True, joblib.load(p)

    def put(self, key: str, value: Any):
        p = self._path(key)
        lock_path = str(p) + ".lock"
        ctx = FileLock(lock_path) if self.lock else nullcontext()
        with ctx:
            data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            if len(data) <= self.small_file:
                with p.open("wb") as fh:
                    fh.write(data)
            else:
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
        "cache",
        "_hash",
        "_raw",
        "_lock",
        "__dict__",
        "__weakref__",
    )

    _hash: int
    _lock: threading.Lock

    @functools.cached_property
    def bound_args(self) -> Mapping[str, Any]:
        """Arguments bound to parameter names."""
        sig = getattr(self.fn, "_node_sig", inspect.signature(self.fn))
        return sig.bind_partial(*self.args, **self.kwargs).arguments

    def __init__(
        self,
        fn,
        args: Tuple = (),
        kwargs: Dict | None = None,
        *,
        flow: "Flow" | None = None,
        cache: bool = True,
    ):
        self.fn = fn
        self.args = tuple(args)
        self.kwargs = kwargs or {}
        self.flow = flow
        self.cache = cache

        self.deps: List[Node] = [
            *(a for a in self.args if isinstance(a, Node)),
            *(v for v in self.kwargs.values() if isinstance(v, Node)),
        ]

        if any(d is self for d in self.deps):
            raise ValueError("Cycle detected in DAG")

        raw = (
            self.fn.__qualname__,
            _canonical_args(self),
        )
        self._raw = raw
        _hash = hashlib.blake2b(repr(raw).encode(), digest_size=6).hexdigest()
        self._hash = int(_hash, 16)
        self._lock = threading.Lock()

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
                    bound=node.bound_args,
                )
                lines.append((node._hash, f"{node.key} = {call}"))
            return lines

    @functools.cached_property
    def order(self) -> List["Node"]:
        order, _ = _build_graph(self, None)
        return order

    @functools.cached_property
    def signature(self) -> str:
        return "\n".join(line for _, line in self.lines)

    def get(self):
        return self._require_flow().run(self, cache_root=self.cache)

    def delete_cache(self) -> None:
        self.delete()

    def generate(self) -> None:
        """Compute and cache this node without returning the value."""
        self._require_flow().generate(self)

    def create(self):
        """Recompute this node ignoring existing cache."""
        return self._require_flow().create(self)

    def delete(self) -> None:
        """Delete cached value for this node."""
        self._require_flow().delete(self)


# ----------------------------------------------------------------------
# DAG helpers
# ----------------------------------------------------------------------
def _build_graph(
    root: Node, cache: Cache | None
) -> tuple[list[Node], dict[Node, list[Node]]]:
    """Return topological order and dependency graph.

    When ``cache`` is provided, traversal stops at nodes with a cache hit,
    effectively pruning their ancestors from the resulting graph.
    """

    edges: dict[Node, list[Node]] = {}
    stack = [root]
    seen: set[Node] = set()
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        hit = False
        if cache is not None and node.cache:
            hit, _ = cache.get(node.key)
        if hit:
            edges[node] = []
            continue
        deps = sorted(node.deps)
        edges[node] = deps
        stack.extend(deps)
    order = list(TopologicalSorter(edges).static_order())
    return order, edges


def _render_call(
    fn: Callable,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    *,
    canonical: bool = False,
    mapping: Dict[Node, str] | None = None,
    ignore: Sequence[str] | None = None,
    bound: Mapping[str, Any] | None = None,
) -> str:
    """Render a function call with argument names.

    ``bound`` may provide a pre-bound argument mapping to skip signature
    inspection.
    """

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
        tuple(sorted((d.key, v) for d, v in mapping.items())) if mapping else None,
        tuple(sorted(ignore or [])),
    )
    with _ren_lock:
        res = _render_cache.get(key)
    if res is not None:
        return res

    sig = getattr(fn, "_node_sig", inspect.signature(fn))
    bound_map = (
        bound if bound is not None else sig.bind_partial(*args, **kwargs).arguments
    )
    ignore_set = set(ignore or ())
    parts = [f"{k}={render(v)}" for k, v in bound_map.items() if k not in ignore_set]
    res = f"{fn.__name__}({', '.join(parts)})"

    with _ren_lock:
        _render_cache[key] = res
    return res


def _call_fn(
    fn: Callable,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    node_key: str | None = None,
) -> Any:
    from .reporters import _track_ctx

    prev = getattr(_track_ctx, "node", None)
    if node_key is not None:
        _track_ctx.node = node_key
    try:
        return fn(*args, **kwargs)
    finally:
        if node_key is not None:
            _track_ctx.node = prev


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
        on_node_end: Callable[[Node, float, bool, bool], None] | None = None,
        on_flow_end: Callable[[Node, float, int, int], None] | None = None,
        continue_on_error: bool = False,
    ):
        self.cache = cache or ChainCache([MemoryLRU(), DiskJoblib()])
        self.executor = executor
        self.workers = workers or (os.cpu_count() or 4)
        self.on_node_start = on_node_start
        self.on_node_end = on_node_end
        self.on_flow_end = on_flow_end
        self.continue_on_error = continue_on_error
        self._can_save = hasattr(self.cache, "save_script")

        self._exec_count = 0
        self._failed: set[str] = set()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def _resolve(self, v):
        if isinstance(v, Node):
            hit, val = self.cache.get(v.key)
            return val if hit else None
        return v

    def _fail_node(
        self, n: Node, start: float, sem: threading.Semaphore | None
    ) -> None:
        if self.on_node_start is not None:
            self.on_node_start(n)
        dur = time.perf_counter() - start
        if self.on_node_end is not None:
            self.on_node_end(n, dur, False, True)
        self._failed.add(n.key)
        if sem is not None:
            sem.release()

    def _eval_node(self, n: Node, sem: threading.Semaphore | None = None):
        if sem is not None:
            sem.acquire()
        start = time.perf_counter()
        if any(d.key in self._failed for d in n.deps):
            self._fail_node(n, start, sem)
            return None
        if self.on_node_start is not None:
            self.on_node_start(n)
        hit, val = self.cache.get(n.key) if n.cache else (False, None)
        if hit:
            dur = time.perf_counter() - start
            if self.on_node_end is not None:
                self.on_node_end(n, dur, True, False)
            if sem is not None:
                sem.release()
            return val

        args = [self._resolve(a) for a in n.args]
        kwargs = {k: self._resolve(v) for k, v in n.kwargs.items()}
        try:
            val = n.fn(*args, **kwargs)
        except Exception as e:  # pragma: no cover - exercised via tests
            if self.continue_on_error:
                logger.error(f"node {n.key} failed for {e}", exc_info=True)
                self._failed.add(n.key)
                dur = time.perf_counter() - start
                if self.on_node_end is not None:
                    self.on_node_end(n, dur, False, True)
                if sem is not None:
                    sem.release()
                return None
            else:
                if sem is not None:
                    sem.release()
                raise
        else:
            self.cache.put(n.key, val)
            if self._can_save:
                self.cache.save_script(n)
        dur = time.perf_counter() - start
        if self.on_node_end is not None:
            self.on_node_end(n, dur, False, False)
        with self._lock:
            self._exec_count += 1
        if sem is not None:
            sem.release()
        return val

    # ------------------------------------------------------------------
    def run(self, root: Node, *, cache_root: bool = True):
        self._exec_count = 0
        self._failed.clear()

        t0 = time.perf_counter()

        use_cache = cache_root and root.cache
        hit, val = self.cache.get(root.key) if use_cache else (False, None)

        if hit:
            if self.on_node_start is not None:
                self.on_node_start(root)
            if self.on_node_end is not None:
                self.on_node_end(root, time.perf_counter() - t0, True, False)
            if self.on_flow_end is not None:
                self.on_flow_end(root, time.perf_counter() - t0, 0, 0)
            return val

        t0 = time.perf_counter()
        order, edges = _build_graph(root, self.cache)

        max_node_workers = 1
        for node in order:
            workers = getattr(node.fn, "_node_workers", 1)
            if workers == -1:
                workers = os.cpu_count() or 1
            if workers > max_node_workers:
                max_node_workers = workers

        if min(self.workers, max_node_workers) <= 1:
            for node in order:
                self._eval_node(node)
            wall = time.perf_counter() - t0
            if self.on_flow_end is not None:
                self.on_flow_end(root, wall, self._exec_count, len(self._failed))
            result = self.cache.get(root.key)[1]

            if not (cache_root and root.cache):
                self.cache.delete(root.key)
            for n in order:
                if not n.cache and n is not root:
                    self.cache.delete(n.key)

            return result

        orig_start = self.on_node_start
        orig_end = self.on_node_end

        def start_cb(n):
            if orig_start is not None:
                orig_start(n)

        def end_cb(n, dur, cached, failed):
            if orig_end is not None:
                orig_end(n, dur, cached, failed)

        self.on_node_start = start_cb
        self.on_node_end = end_cb

        pool_cls = (
            ThreadPoolExecutor if self.executor == "thread" else ProcessPoolExecutor
        )
        proc_q = None
        pool_kwargs: Dict[str, Any] = {"max_workers": self.workers}
        if self.executor == "process":
            from multiprocessing import Queue
            from .reporters import _set_process_queue, _worker_init

            proc_q = Queue()
            _set_process_queue(proc_q)
            pool_kwargs["initializer"] = _worker_init
            pool_kwargs["initargs"] = (proc_q,)
        ts = TopologicalSorter(edges)
        ts.prepare()

        sems: Dict[Callable[..., Any], threading.Semaphore] = {}
        for node in order:
            workers = getattr(node.fn, "_node_workers", 1)
            if workers == -1:
                workers = os.cpu_count() or 1
            if node.fn not in sems:
                sems[node.fn] = threading.Semaphore(workers)

        fut_map = {}
        with pool_cls(**pool_kwargs) as pool:

            def submit(node):
                if any(d.key in self._failed for d in node.deps):
                    self._fail_node(node, time.perf_counter(), None)
                    ts.done(node)
                    for ready in ts.get_ready():
                        submit(ready)
                    return
                if getattr(node.fn, "_node_local", False):
                    self._eval_node(node)
                    ts.done(node)
                    for ready in ts.get_ready():
                        submit(ready)
                    return
                if self.executor == "thread":
                    sem = sems[node.fn]
                    fut_map[pool.submit(self._eval_node, node, sem)] = node
                else:
                    start = time.perf_counter()
                    if self.on_node_start:
                        self.on_node_start(node)
                    hit, val = self.cache.get(node.key) if node.cache else (False, None)
                    if hit:
                        if self.on_node_end:
                            self.on_node_end(
                                node, time.perf_counter() - start, True, False
                            )
                        ts.done(node)
                        return
                    args = [self._resolve(a) for a in node.args]
                    kwargs = {k: self._resolve(v) for k, v in node.kwargs.items()}
                    sem = sems[node.fn]
                    sem.acquire()
                    fut = pool.submit(_call_fn, node.fn, args, kwargs, node.key)
                    fut_map[fut] = (node, start, sem, args, kwargs)

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
                        node, start, sem, args, kwargs = info
                        try:
                            val = fut.result()
                        except (pickle.PicklingError, AttributeError):
                            val = _call_fn(node.fn, args, kwargs, node.key)
                        except Exception as e:
                            if self.continue_on_error:
                                logger.error(
                                    f"node {node.key} failed for {e}",
                                    exc_info=True,
                                )
                                self._failed.add(node.key)
                                dur = time.perf_counter() - start
                                if self.on_node_end is not None:
                                    self.on_node_end(node, dur, False, True)
                                sem.release()
                                ts.done(node)
                                for ready in ts.get_ready():
                                    submit(ready)
                                continue
                            else:
                                sem.release()
                                raise
                        else:
                            self.cache.put(node.key, val)
                            if self._can_save:
                                self.cache.save_script(node)
                        dur = time.perf_counter() - start
                        if self.on_node_end is not None:
                            self.on_node_end(node, dur, False, False)
                        with self._lock:
                            self._exec_count += 1
                        sem.release()
                    ts.done(node)
                for n in ts.get_ready():
                    submit(n)

        wall = time.perf_counter() - t0
        if proc_q is not None:
            _set_process_queue(None)
        self.on_node_start = orig_start
        self.on_node_end = orig_end
        if self.on_flow_end is not None:
            self.on_flow_end(root, wall, self._exec_count, len(self._failed))
        result = self.cache.get(root.key)[1]

        if not (cache_root and root.cache):
            self.cache.delete(root.key)
        for n in order:
            if not n.cache and n is not root:
                self.cache.delete(n.key)

        return result


# ----------------------------------------------------------------------
# public Flow facade
# ----------------------------------------------------------------------


class Config:
    """Store default arguments for tasks using OmegaConf.

    Configuration values may reference other nodes using ``${...}`` syntax. When
    such references are encountered, :class:`Config` will lazily build the
    referenced node with the provided :class:`Flow` instance.
    """

    def __init__(
        self,
        mapping: Mapping[str, Dict[str, Any]] | DictConfig | str | Path | None = None,
        *,
        cache_nodes: bool = False,
    ) -> None:
        """Create a configuration mapping.

        Parameters
        ----------
        mapping:
            Initial configuration data.
        cache_nodes:
            When ``True`` reuse nodes built from this config to avoid repeated
            instantiation. Defaults to ``False``.
        """
        if isinstance(mapping, (str, Path)):
            self._conf: DictConfig = OmegaConf.load(str(mapping))
        else:
            self._conf = OmegaConf.create(mapping or {})
        self._cache_nodes = cache_nodes
        self._nodes: Dict[str, "Node"] = {}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Create Config from a YAML file."""
        return cls(OmegaConf.load(str(path)))

    def _locate(self, path: str) -> Callable[..., Any]:
        mod_name, attr = path.rsplit(".", 1)
        mod = __import__(mod_name, fromlist=[attr])
        return getattr(mod, attr)

    def _resolve_value(self, val: Any, flow: "Flow") -> Any:
        if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
            key = val[2:-1]
            if key in self._conf:
                cfg_val = self._conf[key]
                if OmegaConf.is_dict(cfg_val) and "_target_" in cfg_val:
                    return self._build_node(key, flow)
            return OmegaConf.select(self._conf, key)
        return val

    def _build_node(self, name: str, flow: "Flow") -> "Node":
        if self._cache_nodes and name in self._nodes:
            return self._nodes[name]
        cfg = self._conf[name]
        target = cfg.get("_target_", name)
        if isinstance(target, str) and target.startswith("${") and target.endswith("}"):
            fn_node = self._build_node(target[2:-1], flow)
            fn = fn_node.fn
        else:
            fn = self._locate(str(target))
        params = {
            k: self._resolve_value(v, flow)
            for k, v in OmegaConf.to_container(cfg, resolve=False).items()
            if k != "_target_"
        }
        node = fn(**params)
        if self._cache_nodes:
            self._nodes[name] = node
        return node

    def defaults(self, fn_name: str, *, flow: "Flow" | None = None) -> Dict[str, Any]:
        node_cfg = self._conf.get(fn_name)
        if node_cfg is None:
            return {}
        if flow is None:
            return cast(Dict[str, Any], OmegaConf.to_container(node_cfg, resolve=True))
        result: Dict[str, Any] = {}
        for k, v in OmegaConf.to_container(node_cfg, resolve=False).items():
            if k == "_target_":
                continue
            result[k] = self._resolve_value(v, flow)
        return result

    def copy_from(self, other: "Config") -> None:
        """Copy ``other`` into this config without changing object identity."""
        self._cache_nodes = other._cache_nodes
        self._nodes.clear()
        for key in list(self._conf.keys()):
            del self._conf[key]
        data = OmegaConf.to_container(other._conf, resolve=False) or {}
        for key, value in data.items():
            self._conf[key] = value


class Flow:
    def __init__(
        self,
        *,
        config: Config | None = None,
        cache: Cache | None = None,
        executor: str = "thread",
        default_workers: int = 4,
        reporter: Optional[Any] = None,
        continue_on_error: bool = True,
        validate: bool = True,
    ):
        self.config = config if isinstance(config, Config) else Config(config)
        self._initial_config = Config(
            OmegaConf.create(self.config._conf),
            cache_nodes=self.config._cache_nodes,
        )
        self.default_workers = default_workers
        self.engine = Engine(
            cache=cache,
            executor=executor,
            workers=default_workers,
            continue_on_error=continue_on_error,
        )
        self.validate = validate
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

    def reset_config(self) -> None:
        """Restore the configuration used at initialization."""
        self.config.copy_from(self._initial_config)

    def node(
        self,
        *,
        ignore: Sequence[str] | None = None,
        workers: int | None = None,
        cache: bool = True,
        local: bool = False,
    ) -> Callable[[Callable[..., Any]], Callable[..., Node]]:
        """Decorate ``fn`` into a :class:`Node`.

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

        ignore_set = set(ignore or [])

        def deco(fn: Callable[..., Any]) -> Callable[..., Node]:
            setattr(fn, "_node_ignore", ignore_set)  # type: ignore[attr-defined]
            sig_obj = inspect.signature(fn)
            setattr(fn, "_node_sig", sig_obj)
            setattr(
                fn,
                "_node_workers",
                workers if workers is not None else self.default_workers,
            )
            setattr(fn, "_node_local", local)
            wrapped = (
                validate_arguments(fn, config={"arbitrary_types_allowed": True})
                if self.validate
                else fn
            )
            setattr(wrapped, "_node_ignore", ignore_set)  # type: ignore[attr-defined]
            setattr(wrapped, "_node_sig", sig_obj)
            setattr(wrapped, "_node_workers", getattr(fn, "_node_workers"))
            setattr(wrapped, "_node_local", getattr(fn, "_node_local"))

            @functools.wraps(fn)
            def wrapper(*args, **kwargs) -> Node:
                bound = sig_obj.bind_partial(*args, **kwargs)
                for name, val in self.config.defaults(fn.__name__, flow=self).items():
                    if name not in bound.arguments:
                        bound.arguments[name] = val
                bound.apply_defaults()

                node = Node(wrapped, bound.args, bound.kwargs, flow=self, cache=cache)
                cached = self._registry.get(node)
                if cached is not None:
                    return cached
                self._registry[node] = node
                return node

            wrapper.__signature__ = sig_obj  # type: ignore[attr-defined]
            return wrapper

        return deco

    task = node

    def run(self, root: Node, *, reporter=None, cache_root: bool = True):
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
            return self.engine.run(root, cache_root=cache_root)
        with reporter.attach(self.engine, root):
            return self.engine.run(root, cache_root=cache_root)

    def generate(self, root: Node) -> None:
        """Compute and cache ``root`` without returning the value."""
        self.engine.run(root)
        if isinstance(self.engine.cache, MemoryLRU):
            for n in root.order:
                self.engine.cache.delete(n.key)
        elif isinstance(self.engine.cache, ChainCache):
            for c in self.engine.cache.caches:
                if isinstance(c, MemoryLRU):
                    for n in root.order:
                        c.delete(n.key)

    def create(self, root: Node):
        """Recompute ``root`` ignoring any existing cache."""
        for n in root.order:
            self.engine.cache.delete(n.key)
        return self.run(root)

    def delete(self, root: Node) -> None:
        """Delete cache entry for ``root``."""
        self.engine.cache.delete(root.key)

    def clear_caches(self) -> None:
        """Clear global helper caches."""
        with _can_lock:
            _canonical_cache.clear()
        with _ren_lock:
            _render_cache.clear()


def gather(
    *nodes: Node | Iterable[Node],
    workers: int | None = None,
    cache: bool = True,
) -> Node:
    """Aggregate multiple nodes into a single list result.

    ``nodes`` may be passed either as positional arguments or as a single
    iterable.  All input nodes must belong to the same :class:`Flow`. The
    returned node produces a list of each input node's value in the provided
    order.  ``workers`` controls the concurrent executions of the gather
    node itself.
    """

    if len(nodes) == 1 and not isinstance(nodes[0], Node):
        nodes_list = tuple(cast(Iterable[Node], nodes[0]))
    else:
        nodes_list = cast(Tuple[Node, ...], nodes)

    if not nodes_list:
        raise ValueError("no nodes provided")

    flow = nodes_list[0]._require_flow()
    if any(n.flow is not flow for n in nodes_list):
        raise ValueError("nodes belong to different Flow instances")

    @flow.node(workers=workers, cache=cache)
    def _gather(*items):
        return list(items)

    return _gather(*nodes_list)
