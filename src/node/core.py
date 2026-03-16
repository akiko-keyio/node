"""Core Node definition and DAG graph utilities."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, TYPE_CHECKING
import functools
import inspect
import warnings

import numpy as np
from pydantic import validate_call

if TYPE_CHECKING:
    from .cache import Cache

__all__ = [
    "Node",
    "dimension",
    "define",
    "build_graph",
]

_EMPTY_FROZENSET: frozenset[str] = frozenset()


def _render_call(
    fn: Callable,
    inputs: Mapping[str, Any],
    *,
    mapping: dict["Node", str] | None = None,
    ignore: Sequence[str] | None = None,
) -> str:
    """Render a function call as a string."""

    def render(v: Any) -> str:
        if isinstance(v, Node):
            if mapping and v in mapping:
                return mapping[v]
            return _render_call(
                v.fn,
                v.inputs,
                mapping=mapping,
                ignore=getattr(v.fn, "_node_ignore", ()),
            )
        try:
            v._node_sig
            return v.__name__
        except AttributeError:
            pass
        if isinstance(v, (list, tuple)):
            inner = ", ".join(render(item) for item in v)
            return f"[{inner}]" if isinstance(v, list) else f"({inner})"
        return _canonical(v)

    ignore_set = set(ignore or ())
    parts = [f"{k}={render(v)}" for k, v in inputs.items() if k not in ignore_set]
    return f"{fn.__name__}({', '.join(parts)})"


def _resolve_broadcast(
    fn: Callable,
    inputs: dict[str, Any],
    cache: bool,
    reduce_dims: Sequence[str] | str,
) -> tuple[tuple[str, ...], dict[str, list[Any]], np.ndarray | None]:
    """Compute dims/coords/_items for a Node if any input is a vector Node.

    Returns ``(dims, coords, items)`` where *items* is ``None`` when no
    broadcasting applies (scalar node or explicit dims passed by caller).
    """
    vector_inputs = {
        k: v for k, v in inputs.items() if isinstance(v, Node) and v.dims
    }
    if not vector_inputs:
        return (), {}, None

    if reduce_dims:
        ordered = tuple(
            dict.fromkeys(d for v in vector_inputs.values() for d in v.dims)
        )
        all_dims = set(ordered)

        if reduce_dims == "all":
            reduce_set = all_dims
            reduce_order = ordered
        else:
            reduce_order = (
                (reduce_dims,) if isinstance(reduce_dims, str) else tuple(reduce_dims)
            )
            reduce_set = set(reduce_order)
            invalid = reduce_set - all_dims
            if invalid:
                from .exceptions import DimensionMismatchError

                raise DimensionMismatchError(
                    f"reduce_dims {invalid} not in input dims {all_dims}"
                )

        output_dims = tuple(d for d in sorted(all_dims) if d not in reduce_set)
        return _broadcast(
            fn, inputs, vector_inputs, cache,
            target_dims=output_dims, reduce_dims_order=reduce_order,
        )

    return _broadcast(fn, inputs, vector_inputs, cache, target_dims=None)


class Node:
    """Represents a computation unit in the DAG."""

    __slots__ = (
        "fn",
        "inputs",
        "deps_nodes",
        "_exec_deps",
        "cache",
        "_hash",
        "dims",
        "coords",
        "_items",
        "__weakref__",
        "_order",
        "_script_lines",
        "_script",
    )

    _hash: int

    def __init__(
        self,
        fn: Callable,
        inputs: dict[str, Any],
        *,
        cache: bool = True,
        dims: tuple[str, ...] = (),
        coords: dict[str, list[Any]] | None = None,
        reduce_dims: Sequence[str] | str = (),
        _deps: list[Node] | None = None,
    ):
        self.fn = fn
        self.inputs = inputs
        self.cache = bool(cache)
        self._items: np.ndarray | None = None

        if not dims:
            self.dims, self.coords, self._items = _resolve_broadcast(
                fn, inputs, cache, reduce_dims
            )
            if self._items is None:
                self.dims = dims
                self.coords = coords or {}
        else:
            self.dims = dims
            self.coords = coords or {}

        self.deps_nodes = _deps if _deps is not None else _collect_nodes(self.inputs.values())

        self._exec_deps: list[Node] = list(self.deps_nodes)
        if self._items is not None:
            self._exec_deps.extend(self._items.flat)

        self._hash = compute_node_hash(
            fn.__name__, inputs, getattr(fn, "_node_ignore", _EMPTY_FROZENSET)
        )

    # -- Serialization --------------------------------------------------------

    def __getstate__(self):
        exclude = {"__weakref__", "_order", "_script", "_script_lines"}
        return {
            k: getattr(self, k)
            for k in self.__slots__
            if k not in exclude and hasattr(self, k)
        }

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    # -- Identity -------------------------------------------------------------

    def __repr__(self):
        return self.script

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Node) and self._hash == other._hash

    def __hash__(self) -> int:
        return self._hash

    # -- Script generation (lazy) ---------------------------------------------

    @property
    def script_lines(self) -> list[tuple[int, str]]:
        try:
            return self._script_lines
        except AttributeError:
            pass

        from graphlib import TopologicalSorter
        from collections import defaultdict

        edges: dict[Node, list[Node]] = {}
        stack = [self]
        seen: set[int] = set()
        while stack:
            n = stack.pop()
            if n._hash in seen:
                continue
            seen.add(n._hash)
            edges[n] = n.deps_nodes
            stack.extend(n.deps_nodes)

        order = list(TopologicalSorter(edges).static_order())

        counters: dict[str, int] = defaultdict(int)
        names: dict[Node, str] = {}
        for n in order:
            name = n.fn.__name__
            names[n] = f"{name}_{counters[name]}"
            counters[name] += 1

        lines: list[tuple[int, str]] = []
        for n in order:
            var_map = {d: names[d] for d in n.deps_nodes}
            ig = getattr(n.fn, "_node_ignore", ())
            call = _render_call(n.fn, n.inputs, mapping=var_map, ignore=ig)
            if n.dims:
                dc = ", ".join(
                    f"{d}:{len(n.coords.get(d, []))}" for d in n.dims
                )
                line = f"{names[n]} = {call}  # dims=({dc})"
            else:
                line = f"{names[n]} = {call}"
            lines.append((n._hash, line))

        self._script_lines = lines
        return lines

    @property
    def order(self) -> list[Node]:
        try:
            return self._order
        except AttributeError:
            self._order, _ = build_graph(self, None)
            return self._order

    @property
    def script(self) -> str:
        try:
            return self._script
        except AttributeError:
            pass
        body = "\n".join(line for _, line in self.script_lines)
        self._script = f"# hash = {self._hash:x}\n{body}"
        return self._script

    # -- Execution ------------------------------------------------------------

    def __call__(self, *, force: bool = False):
        from .runtime import get_runtime

        if force:
            self.invalidate()
        return get_runtime().run(self, cache_root=self.cache)

    def invalidate(self, *, recursive: bool = False) -> None:
        """Clear cache for this node. If *recursive* is true, also clear cache for all dependencies."""
        from .runtime import get_runtime

        get_runtime().delete(self, recursive=recursive)


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _collect_nodes(values) -> list[Node]:
    """Collect all Node objects reachable from a nested value structure."""
    result: list[Node] = []
    stack: list[Any] = list(values)
    stack.reverse()

    while stack:
        item = stack.pop()

        if isinstance(item, Node):
            result.append(item)
            continue
        if isinstance(item, (str, bytes)):
            continue
        if isinstance(item, Mapping):
            stack.extend(reversed(list(item.values())))
            continue
        if not isinstance(item, Iterable):
            continue
        if getattr(item, "ndim", 1) == 0:
            continue

        if isinstance(item, np.ndarray):
            if item.dtype != np.object_ or item.size == 0:
                continue
            for x in item.flat:
                if isinstance(x, Node):
                    result.append(x)
            continue

        children = list(item)
        stack.extend(reversed(children))

    return result


# Late imports
from .dimension import dimension, broadcast as _broadcast
from .hashing import compute_node_hash, _canonical
from .cache_namespace import cache_namespace


def build_graph(
    root: Node,
    cache: "Cache | None",
    *,
    cache_filter: Callable[[Node, bool], bool] | None = None,
) -> tuple[list[Node], dict[Node, list[Node]]]:
    """Return topological order and dependency edges.

    When *cache* is provided, nodes with a cache hit are treated as leaves
    (their ancestors are pruned from the graph).
    """
    edges: dict[Node, list[Node]] = {}
    order: list[Node] = []
    seen: set[Node] = set()
    stack: list[tuple[Node, bool]] = [(root, False)]

    while stack:
        node, expanded = stack.pop()
        if expanded:
            order.append(node)
            continue
        if node in seen:
            continue
        seen.add(node)

        hit = (
            cache is not None
            and node.cache
            and (
                cache_filter(node, node is root)
                if cache_filter is not None
                else True
            )
            and cache._has_entry(cache_namespace(node), node._hash)
        )
        if hit:
            edges[node] = []
            order.append(node)
            continue

        deps = node._exec_deps
        edges[node] = deps
        stack.append((node, True))
        for dep in reversed(deps):
            if dep not in seen:
                stack.append((dep, False))

    return order, edges


def define(
    *,
    ignore: list[str] | None = None,
    workers: int | None = None,
    cache: bool = True,
    local: bool = False,
    reduce_dims: Sequence[str] | str = (),
) -> Callable[[Callable[..., Any]], Callable[..., Node]]:
    """Decorate a function to create a Node factory.

    Parameters
    ----------
    ignore : list[str], optional
        Argument names excluded from the cache key.
    workers : int, optional
        Maximum concurrency for this function. ``-1`` uses all cores.
    cache : bool, optional
        Whether to cache results. Defaults to True.
    local : bool, optional
        Execute in the caller thread. Defaults to False.
    reduce_dims : Sequence[str] | str, optional
        Dimensions to reduce over when aggregating.
    """
    ignore_frozen = frozenset(ignore or [])

    def deco(fn: Callable[..., Any]) -> Callable[..., Node]:
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
            "_node_ignore": ignore_frozen,
            "_node_sig": sig_obj,
            "_node_local": local,
        }
        for k, v in node_attrs.items():
            setattr(fn, k, v)

        fillable_params = frozenset(
            name
            for name, param in sig_obj.parameters.items()
            if param.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        )

        validated_fn: Callable[..., Any] | None = None

        def get_validated_fn() -> Callable[..., Any]:
            nonlocal validated_fn
            if validated_fn is None:
                validated_fn = validate_call(
                    fn, config={"arbitrary_types_allowed": True}
                )
                for k, v in node_attrs.items():
                    setattr(validated_fn, k, v)
            return validated_fn

        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Node:
            from .runtime import get_runtime

            rt = get_runtime()
            selected_fn = get_validated_fn() if rt.validate else fn
            setattr(
                selected_fn,
                "_node_workers",
                workers if workers is not None else rt.workers,
            )
            bound = sig_obj.bind_partial(*args, **kwargs)

            missing = fillable_params - bound.arguments.keys()
            if missing:
                for name, val in rt.config.defaults(
                    fn.__name__, runtime=rt, selected_names=missing
                ).items():
                    if name not in bound.arguments:
                        bound.arguments[name] = val
            bound.apply_defaults()

            node = Node(
                selected_fn,
                bound.arguments,
                cache=cache,
                reduce_dims=reduce_dims,
            )
            cached = rt._registry.get(node)
            if cached is not None:
                return cached
            rt._registry[node] = node
            return node

        wrapper.__signature__ = sig_obj  # type: ignore[attr-defined]
        return wrapper

    return deco
