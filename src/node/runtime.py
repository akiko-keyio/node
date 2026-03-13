"""Singleton Runtime for executing DAG nodes."""

from __future__ import annotations

import inspect
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import nullcontext

from graphlib import TopologicalSorter
from collections.abc import Callable, Sequence
from typing import Any, TYPE_CHECKING, Literal
from weakref import WeakValueDictionary
from omegaconf import OmegaConf
import numpy as np

from .cache import Cache, ChainCache, DiskCache, MemoryLRU
from .cache_namespace import cache_namespace
from .config import Config
from .core import build_graph
from .logger import logger
from .parallel import parallel_context

if TYPE_CHECKING:
    from .core import Node

__all__ = [
    "Runtime",
    "get_runtime",
    "configure",
]

# Singleton instance
_runtime: "Runtime" | None = None
_runtime_lock = threading.Lock()


def get_runtime() -> "Runtime":
    """Get the global Runtime instance."""
    global _runtime
    if _runtime is None:
        raise RuntimeError("Runtime is not configured. Call configure() first.")
    return _runtime


def configure(
    *,
    config: Config | str | None = None,
    cache: Cache | None = None,
    executor: Literal["thread"] = "thread",
    workers: int = 4,
    reporter: Any = None,
    continue_on_error: bool = True,
    validate: bool = True,
    limit_inner_parallelism: bool = True,
) -> "Runtime":
    """Configure the global Runtime. Can only be called once.

    Parameters
    ----------
    config:
        Configuration object or path to YAML file.
    cache:
        Cache instance. Defaults to ChainCache(MemoryLRU, DiskCache).
    executor:
        Execution backend. Only ``"thread"`` is supported.
    workers:
        Default worker count.
    reporter:
        Progress reporter instance.
    continue_on_error:
        If True, continue execution when nodes fail.
    validate:
        If True, use pydantic to validate function arguments.
    limit_inner_parallelism:
        If True, automatically limit inner parallelism (e.g., sklearn n_jobs,
        numpy BLAS threads) to prevent thread explosion. Defaults to True.
    Returns
    -------
    Runtime
        The configured Runtime instance.
    """
    global _runtime
    if _runtime is not None:
        raise RuntimeError("Runtime already configured. Call reset() first if needed.")

    with _runtime_lock:
        _runtime = Runtime(
            config=config,
            cache=cache,
            executor=executor,
            workers=workers,
            reporter=reporter,
            continue_on_error=continue_on_error,
            validate=validate,
            limit_inner_parallelism=limit_inner_parallelism,
        )
    return _runtime


def reset() -> None:
    """Reset the global Runtime. Primarily for testing."""
    global _runtime
    with _runtime_lock:
        _runtime = None


class Runtime:
    """Singleton runtime for executing DAG nodes.

    Manages configuration, caching, and execution mode for the entire
    application lifecycle.
    """

    def __init__(
        self,
        *,
        config: Config | str | None = None,
        cache: Cache | None = None,
        executor: Literal["thread"] = "thread",
        workers: int = 4,
        reporter: Any = None,
        continue_on_error: bool = True,
        validate: bool = True,
        limit_inner_parallelism: bool = True,
    ):
        # Config
        self.config = config if isinstance(config, Config) else Config(config)

        self._initial_config = Config(
            OmegaConf.create(self.config._conf),
        )

        # Cache
        self.cache = cache or ChainCache([MemoryLRU(), DiskCache()])

        # Execution
        if executor != "thread":
            raise ValueError("Only executor='thread' is supported.")
        self.executor: Literal["thread"] = "thread"
        self.workers = workers
        self.continue_on_error = continue_on_error
        self.limit_inner_parallelism = limit_inner_parallelism
        self.validate = validate

        # Internal state
        self._can_save = hasattr(self.cache, "save_script")
        self._exec_count = 0
        self._failed: set[str] = set()
        self._lock = threading.Lock()
        self._registry: WeakValueDictionary["Node", "Node"] = WeakValueDictionary()
        self._results: dict[int, Any] = {}
        # Reporter
        if reporter is None:
            try:
                from .reporter import RichReporter
                self.reporter = RichReporter()
            except Exception:
                self.reporter = None
        else:
            self.reporter = reporter

        # Callbacks (for reporter integration)
        self.on_node_start: Callable[[Node], None] | None = None
        self.on_node_end: Callable[[Node, float, bool, bool], None] | None = None
        self.on_node_state: Callable[[Node, str], None] | None = None
        self.on_flow_end: Callable[[Node, float, int, int], None] | None = None


    def reset_config(self) -> None:
        """Restore the configuration used at initialization."""
        self.config.copy_from(self._initial_config)

    def _set_node_state(self, node: Node, state: str) -> None:
        """Emit node state transitions for progress reporters."""
        if self.on_node_state is not None:
            self.on_node_state(node, state)


    def _resolve(self, v):
        """Recursively resolve parameter values, replacing Nodes with their cached results."""
        from .core import Node
        from .dimension import DimensionedResult
        if isinstance(v, Node):
            # First check _results (for just-computed nodes), then cache
            if v._hash in self._results:
                return self._results[v._hash]
            if not self._cache_enabled_for(v):
                return None
            hit, val = self.cache.get(cache_namespace(v), v._hash)
            return val if hit else None
        elif isinstance(v, dict):
            return {k: self._resolve(item) for k, item in v.items()}
        elif isinstance(v, (list, tuple)):
            items = [self._resolve(item) for item in v]
            return tuple(items) if isinstance(v, tuple) else items
        elif isinstance(v, np.ndarray):
            if v.dtype == object:
                # Recursively resolve Nodes inside the array
                # Use explicit loop to prevent NumPy 2.x from unpacking iterables
                res = [self._resolve(item) for item in v.flat]
                out = np.empty(len(res), dtype=object)
                for i, r in enumerate(res):
                    out[i] = r
                resolved = out.reshape(v.shape)
                if isinstance(v, DimensionedResult):
                    return DimensionedResult(
                        resolved,
                        dims=v.dims,
                        coords=v.coords,
                    )
                return resolved
            return v
        return v

    def _cache_enabled_for(self, node: "Node") -> bool:
        """Return whether cache read/write is enabled for this node."""
        return node.cache

    def _fail_node(
        self, n: Node, start: float, sem: threading.Semaphore | None
    ) -> None:
        if self.on_node_start is not None:
            self.on_node_start(n)
        dur = time.perf_counter() - start
        if self.on_node_end is not None:
            self.on_node_end(n, dur, False, True)
        self._failed.add(n._hash)
        if sem is not None:
            sem.release()

    def _bind_args(self, node: Node, resolved_args: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
        """Bind resolved arguments to function signature."""
        sig = getattr(node.fn, "_node_sig", None)
        if sig is None:
            sig = inspect.signature(node.fn)
        args = []
        kwargs = {}
        for name, param in sig.parameters.items():
            if name not in resolved_args:
                continue
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                args.extend(resolved_args[name])
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                kwargs.update(resolved_args[name])
            else:
                kwargs[name] = resolved_args[name]
        return args, kwargs

    def _save_result(
        self, node: Node, val: Any, start: float, *, count_exec: bool = True
    ) -> None:
        """Save result to cache and trigger callbacks after successful execution."""
        self._results[node._hash] = val
        if count_exec:
            with self._lock:
                self._exec_count += 1
        
        if self._cache_enabled_for(node):
            self._set_node_state(node, "cache_writing")
            self.cache.put(cache_namespace(node), node._hash, val)
            if self._can_save:
                self.cache.save_script(node)
        
        dur = time.perf_counter() - start
        if self.on_node_end is not None:
            self.on_node_end(node, dur, False, False)

    def _ensure_deps_ready(self, node: Node) -> None:
        """Ensure dependencies are materialized when a pruned cache hit turns miss."""
        for dep in node._exec_deps:
            if dep._hash in self._results:
                continue
            if self._cache_enabled_for(dep):
                dep_hit, dep_val = self.cache.get(cache_namespace(dep), dep._hash)
                if dep_hit:
                    self._results[dep._hash] = dep_val
                    continue
            # Fallback: evaluate dependency recursively (self-heal for corrupt cache files).
            self._eval_node(dep)

    def _eval_node(self, n: Node, sem: threading.Semaphore | None = None):
        if n._hash in self._results:
            return self._results[n._hash]

        start = time.perf_counter()
        if any(d._hash in self._failed for d in n._exec_deps):
            # Semaphore is acquired only for real computation after cache miss.
            self._fail_node(n, start, None)
            return None
        if self.on_node_start is not None:
            self.on_node_start(n)
        cache_enabled = self._cache_enabled_for(n)
        if cache_enabled:
            self._set_node_state(n, "cache_reading")
        hit, val = (
            self.cache.get(cache_namespace(n), n._hash) if cache_enabled else (False, None)
        )
        if hit:
            dur = time.perf_counter() - start
            self._results[n._hash] = val
            if self.on_node_end is not None:
                self.on_node_end(n, dur, True, False)
            return val

        if n._exec_deps:
            self._set_node_state(n, "waiting")
            self._ensure_deps_ready(n)

        acquired = False
        if sem is not None:
            self._set_node_state(n, "waiting")
            sem.acquire()
            acquired = True

        try:
            self._set_node_state(n, "executing")
            if n._items is not None:
                # VectorNode / Reduction Node:
                # The result is the aggregation of its items (which are dependencies).
                from .dimension import DimensionedResult

                # Items are dependency nodes already materialized by _ensure_deps_ready.
                results = [
                    self._results.get(item._hash) if hasattr(item, "_hash") else item
                    for item in n._items.flat
                ]

                # Reconstruct array structure
                # Use explicit loop assignment to prevent NumPy 2.x from unpacking
                # iterable objects (like DataFrames) during slice assignment
                val_flat = np.empty(len(results), dtype=object)
                for i, res in enumerate(results):
                    val_flat[i] = res
                val_array = val_flat.reshape(n._items.shape)

                # Return as DimensionedResult if has dimensions, else unwrap scalar
                if n.dims:
                    val = DimensionedResult(val_array, dims=n.dims, coords=n.coords)
                else:
                    val = val_array.item()

                self._save_result(n, val, start, count_exec=False)
                return val

            # Resolve arguments from inputs and call function
            resolved_args = {k: self._resolve(v) for k, v in n.inputs.items()}
            args, kwargs = self._bind_args(n, resolved_args)
            try:
                val = n.fn(*args, **kwargs)
            except Exception as e:
                logger.error("node {} failed for {}", f"{n.fn.__name__}_{n._hash:x}", e, exc_info=True)
                if self.continue_on_error:
                    self._failed.add(n._hash)
                    dur = time.perf_counter() - start
                    if self.on_node_end is not None:
                        self.on_node_end(n, dur, False, True)
                    return None
                else:
                    raise
            else:
                self._save_result(n, val, start)
            return val
        finally:
            if acquired:
                sem.release()

    def run(self, root: Node, *, reporter=None, cache_root: bool = True):
        """Run the DAG rooted at ``root``."""
        if reporter is None:
            reporter = self.reporter
        
        # Apply parallel context to limit inner parallelism
        ctx = parallel_context(self.workers) if self.limit_inner_parallelism else nullcontext()
        
        with ctx:
            if reporter is None:
                return self._run_internal(root, cache_root=cache_root)
            graph = build_graph(root, self.cache)
            try:
                attach_sig = inspect.signature(reporter.attach)
                supports_order = "order" in attach_sig.parameters
            except (TypeError, ValueError):
                supports_order = False
            attach_ctx = (
                reporter.attach(self, root, order=graph[0])
                if supports_order
                else reporter.attach(self, root)
            )
            with attach_ctx:
                return self._run_internal(root, cache_root=cache_root, _graph=graph)

    def _run_internal(
        self,
        root: Node,
        *,
        cache_root: bool = True,
        _graph: tuple[list[Node], dict[Node, list[Node]]] | None = None,
    ):
        self._exec_count = 0
        self._failed.clear()
        self._results.clear()

        t0 = time.perf_counter()

        use_cache = cache_root and self._cache_enabled_for(root)
        hit, val = (
            self.cache.get(cache_namespace(root), root._hash)
            if use_cache
            else (False, None)
        )

        if hit:
            self._results[root._hash] = val
            if self.on_node_start is not None:
                self.on_node_start(root)
            self._set_node_state(root, "cache_reading")
            if self.on_node_end is not None:
                self.on_node_end(root, time.perf_counter() - t0, True, False)
            if self.on_flow_end is not None:
                self.on_flow_end(root, time.perf_counter() - t0, 0, 0)
            return val

        if _graph is None:
            order, edges = build_graph(root, self.cache)
        else:
            order, edges = _graph

        # Track how many downstream nodes still need each node's in-run result.
        # Once a node has no remaining consumers, its value can be released
        # from ``_results`` to reduce peak memory usage.
        dep_refcounts: dict[int, int] = {}
        for deps in edges.values():
            for dep in deps:
                dep_refcounts[dep._hash] = dep_refcounts.get(dep._hash, 0) + 1

        def release_consumed_deps(node: Node) -> None:
            for dep in node._exec_deps:
                dep_hash = dep._hash
                remaining = dep_refcounts.get(dep_hash)
                if remaining is None:
                    continue
                remaining -= 1
                if remaining <= 0:
                    dep_refcounts.pop(dep_hash, None)
                    if dep is not root:
                        self._results.pop(dep_hash, None)
                else:
                    dep_refcounts[dep_hash] = remaining

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
                release_consumed_deps(node)
            wall = time.perf_counter() - t0
            if self.on_flow_end is not None:
                self.on_flow_end(root, wall, self._exec_count, len(self._failed))
            
            # Use local results instead of re-getting from cache
            result = self._results.get(root._hash)
            return result

        ts = TopologicalSorter(edges)
        ts.prepare()

        sems: dict[Callable[..., Any], threading.Semaphore] = {}
        for node in order:
            workers = getattr(node.fn, "_node_workers", 1)
            if workers == -1:
                workers = os.cpu_count() or 1
            if node.fn not in sems:
                sems[node.fn] = threading.Semaphore(workers)

        fut_map: dict[Any, Node] = {}
        with ThreadPoolExecutor(max_workers=self.workers) as pool:

            def submit(node):
                if any(d._hash in self._failed for d in node._exec_deps):
                    self._fail_node(node, time.perf_counter(), None)
                    release_consumed_deps(node)
                    ts.done(node)
                    for ready in ts.get_ready():
                        submit(ready)
                    return
                if getattr(node.fn, "_node_local", False):
                    self._eval_node(node)
                    release_consumed_deps(node)
                    ts.done(node)
                    for ready in ts.get_ready():
                        submit(ready)
                    return
                sem = sems[node.fn]
                fut_map[pool.submit(self._eval_node, node, sem)] = node

            for n in ts.get_ready():
                submit(n)

            while fut_map:
                done, _ = wait(fut_map, return_when=FIRST_COMPLETED)
                for fut in done:
                    node = fut_map.pop(fut)
                    fut.result()
                    release_consumed_deps(node)
                    ts.done(node)
                for n in ts.get_ready():
                    submit(n)

        wall = time.perf_counter() - t0
        if self.on_flow_end is not None:
            self.on_flow_end(root, wall, self._exec_count, len(self._failed))
        
        return self._results.get(root._hash)

    def delete(self, root: Node) -> None:
        """Delete cache entries reachable from ``root``.

        Under item-cache-first semantics, invalidating a vector node should
        clear its cached item subgraph as well, otherwise a rerun may still
        hit item caches and appear non-invalidated.
        """
        stack: list[Node] = [root]
        seen: set[int] = set()
        while stack:
            node = stack.pop()
            if node._hash in seen:
                continue
            seen.add(node._hash)
            if node.cache:
                self.cache.delete(cache_namespace(node), node._hash)
            for dep in node._exec_deps:
                if hasattr(dep, "_hash"):
                    stack.append(dep)
