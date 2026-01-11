"""Singleton Runtime for executing DAG nodes."""

from __future__ import annotations

import functools
import inspect
import os
import pickle
import threading
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from graphlib import TopologicalSorter
from collections.abc import Callable, Sequence
from typing import Any
from weakref import WeakValueDictionary

from loky import ProcessPoolExecutor
from omegaconf import OmegaConf
from pydantic import validate_arguments

from .cache import Cache, ChainCache, DiskJoblib, MemoryLRU
from .config import Config
from .core import Node, _build_graph, clear_caches
from .logger import logger

__all__ = [
    "Runtime",
    "get_runtime",
    "configure",
]

# Singleton instance
_runtime: "Runtime" | None = None
_runtime_lock = threading.Lock()


def get_runtime() -> "Runtime":
    """Get the global Runtime instance, creating one with defaults if needed."""
    global _runtime
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                _runtime = Runtime()
    return _runtime


def configure(
    *,
    config: Config | str | None = None,
    cache: Cache | None = None,
    executor: str = "thread",
    workers: int = 4,
    reporter: Any = None,
    continue_on_error: bool = True,
    validate: bool = True,
) -> "Runtime":
    """Configure the global Runtime. Can only be called once.

    Parameters
    ----------
    config:
        Configuration object or path to YAML file.
    cache:
        Cache instance. Defaults to ChainCache(MemoryLRU, DiskJoblib).
    executor:
        "thread" or "process".
    workers:
        Default worker count.
    reporter:
        Progress reporter instance.
    continue_on_error:
        If True, continue execution when nodes fail.
    validate:
        If True, use pydantic to validate function arguments.

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
        )
    return _runtime


def reset() -> None:
    """Reset the global Runtime. Primarily for testing."""
    global _runtime
    with _runtime_lock:
        _runtime = None


def _call_fn(
    fn: Callable,
    args: Sequence[Any],
    kwargs: dict[str, Any],
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
        executor: str = "thread",
        workers: int = 4,
        reporter: Any = None,
        continue_on_error: bool = True,
        validate: bool = True,
    ):
        # Config
        self.config = config if isinstance(config, Config) else Config(config)

        self._initial_config = Config(
            OmegaConf.create(self.config._conf),
            cache_nodes=self.config._cache_nodes,
        )

        # Cache
        self.cache = cache or ChainCache([MemoryLRU(), DiskJoblib()])

        # Execution
        self.executor = executor
        self.workers = workers
        self.continue_on_error = continue_on_error
        self.validate = validate

        # Internal state
        self._can_save = hasattr(self.cache, "save_script")
        self._exec_count = 0
        self._failed: set[str] = set()
        self._lock = threading.Lock()
        self._registry: WeakValueDictionary[Node, Node] = WeakValueDictionary()

        # Reporter
        if reporter is None:
            try:
                from .reporters import RichReporter
                self.reporter = RichReporter()
            except Exception:
                self.reporter = None
        else:
            self.reporter = reporter

        # Callbacks (for reporter integration)
        self.on_node_start: Callable[[Node], None] | None = None
        self.on_node_end: Callable[[Node, float, bool, bool], None] | None = None
        self.on_flow_end: Callable[[Node, float, int, int], None] | None = None


    def reset_config(self) -> None:
        """Restore the configuration used at initialization."""
        self.config.copy_from(self._initial_config)

    def _cleanup_uncached(self, root: Node, order: list[Node], cache_root: bool) -> None:
        """Delete cache entries for nodes that should not be cached."""
        if not (cache_root and root.cache):
            self.cache.delete(root.key)
        for n in order:
            if not n.cache and n is not root:
                self.cache.delete(n.key)

    def define(
        self,
        *,
        ignore: Sequence[str] | None = None,
        workers: int | None = None,
        cache: bool = True,
        local: bool = False,
    ) -> Callable[[Callable[..., Any]], Callable[..., Node]]:
        """Decorate a function to create a Node.

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
            # 检测闭包变量
            if fn.__closure__ is not None:
                warnings.warn(
                    f"Function '{fn.__name__}' uses closure variables. "
                    f"This may violate purity constraints as closure variables "
                    f"are not included in the node's cache key. "
                    f"Consider passing all dependencies as explicit parameters.",
                    category=UserWarning,
                    stacklevel=2,
                )
            
            sig_obj = inspect.signature(fn)
            node_attrs = {
                "_node_ignore": ignore_set,
                "_node_sig": sig_obj,
                "_node_workers": workers if workers is not None else self.workers,
                "_node_local": local,
            }

            for k, v in node_attrs.items():
                setattr(fn, k, v)

            wrapped = (
                validate_arguments(fn, config={"arbitrary_types_allowed": True})
                if self.validate
                else fn
            )

            if wrapped is not fn:
                for k, v in node_attrs.items():
                    setattr(wrapped, k, v)

            @functools.wraps(fn)
            def wrapper(*args, **kwargs) -> Node:
                bound = sig_obj.bind_partial(*args, **kwargs)
                for name, val in self.config.defaults(fn.__name__, runtime=self).items():
                    if name not in bound.arguments:
                        bound.arguments[name] = val
                bound.apply_defaults()

                node = Node(wrapped, bound.arguments, cache=cache, runtime=self)
                cached_node = self._registry.get(node)
                if cached_node is not None:
                    return cached_node
                self._registry[node] = node
                return node

            wrapper.__signature__ = sig_obj  # type: ignore[attr-defined]
            
            # Add sweep method to wrapper for convenient access: process.sweep(...)
            from .core import sweep as core_sweep
            def wrapper_sweep(config, *, workers=None, cache=True, **kwargs):
                return core_sweep(wrapper, config=config, workers=workers, cache=cache, **kwargs)
            wrapper.sweep = wrapper_sweep
            
            return wrapper

        return deco



    def _resolve(self, v):
        """递归解析参数值，将 Node 替换为其缓存的计算结果。"""
        if isinstance(v, Node):
            hit, val = self.cache.get(v.key)
            return val if hit else None
        elif isinstance(v, tuple):
            return tuple(self._resolve(item) for item in v)
        elif isinstance(v, list):
            return [self._resolve(item) for item in v]
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

    def _bind_args(self, node: Node, resolved_args: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
        """Bind resolved arguments to function signature."""
        sig = getattr(node.fn, "_node_sig", inspect.signature(node.fn))
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

    def _save_result(self, node: Node, val: Any, start: float) -> None:
        """Save result to cache and trigger callbacks after successful execution."""
        self.cache.put(node.key, val)
        if self._can_save:
            self.cache.save_script(node)
        dur = time.perf_counter() - start
        if self.on_node_end is not None:
            self.on_node_end(node, dur, False, False)
        with self._lock:
            self._exec_count += 1

    def _eval_node(self, n: Node, sem: threading.Semaphore | None = None):
        if sem is not None:
            sem.acquire()
        start = time.perf_counter()
        if any(d.key in self._failed for d in n.deps_nodes):
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

        # 从 inputs 解析参数并调用函数
        resolved_args = {k: self._resolve(v) for k, v in n.inputs.items()}
        args, kwargs = self._bind_args(n, resolved_args)
        try:
            val = n.fn(*args, **kwargs)
        except Exception as e:
            logger.error("node {} failed for {}", n.key, e, exc_info=True)
            if self.continue_on_error:
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
            self._save_result(n, val, start)
        if sem is not None:
            sem.release()
        return val

    def run(self, root: Node, *, reporter=None, cache_root: bool = True):
        """Run the DAG rooted at ``root``."""
        if reporter is None:
            reporter = self.reporter
        if reporter is None:
            return self._run_internal(root, cache_root=cache_root)
        with reporter.attach(self, root):
            return self._run_internal(root, cache_root=cache_root)

    def _run_internal(self, root: Node, *, cache_root: bool = True):
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
            self._cleanup_uncached(root, order, cache_root)

            return result

        orig_start = self.on_node_start
        orig_end = self.on_node_end

        pool_cls = (
            ThreadPoolExecutor if self.executor == "thread" else ProcessPoolExecutor
        )
        proc_q = None
        pool_kwargs: dict[str, Any] = {"max_workers": self.workers}
        if self.executor == "process":
            from multiprocessing import Queue
            from .reporters import _set_process_queue, _worker_init

            proc_q = Queue()
            _set_process_queue(proc_q)
            pool_kwargs["initializer"] = _worker_init
            pool_kwargs["initargs"] = (proc_q,)
        ts = TopologicalSorter(edges)
        ts.prepare()

        sems: dict[Callable[..., Any], threading.Semaphore] = {}
        for node in order:
            workers = getattr(node.fn, "_node_workers", 1)
            if workers == -1:
                workers = os.cpu_count() or 1
            if node.fn not in sems:
                sems[node.fn] = threading.Semaphore(workers)

        fut_map = {}
        with pool_cls(**pool_kwargs) as pool:

            def submit(node):
                if any(d.key in self._failed for d in node.deps_nodes):
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
                    resolved_bound = {k: self._resolve(v) for k, v in node.inputs.items()}
                    args, kwargs = self._bind_args(node, resolved_bound)
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
                        fut.result()
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
                            self._save_result(node, val, start)
                        sem.release()
                    ts.done(node)
                for n in ts.get_ready():
                    submit(n)

        wall = time.perf_counter() - t0
        if proc_q is not None:
            from .reporters import _set_process_queue
            _set_process_queue(None)
        self.on_node_start = orig_start
        self.on_node_end = orig_end
        if self.on_flow_end is not None:
            self.on_flow_end(root, wall, self._exec_count, len(self._failed))
        result = self.cache.get(root.key)[1]

        self._cleanup_uncached(root, order, cache_root)

        return result

    def generate(self, root: Node) -> None:
        """Compute and cache ``root`` without returning the value."""
        self.run(root)
        if isinstance(self.cache, MemoryLRU):
            for n in root.order:
                self.cache.delete(n.key)
        elif isinstance(self.cache, ChainCache):
            for c in self.cache.caches:
                if isinstance(c, MemoryLRU):
                    for n in root.order:
                        c.delete(n.key)

    def create(self, root: Node):
        """Recompute ``root`` ignoring any existing cache."""
        for n in root.order:
            self.cache.delete(n.key)
        return self.run(root)

    def delete(self, root: Node) -> None:
        """Delete cache entry for ``root``."""
        self.cache.delete(root.key)

    def clear_caches(self) -> None:
        """Clear global helper caches."""
        clear_caches()