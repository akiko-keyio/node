"""Core Node definition and DAG graph utilities."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, TYPE_CHECKING
import functools
import inspect
import warnings
from graphlib import TopologicalSorter

from pydantic import validate_call

if TYPE_CHECKING:
    from .cache import Cache

__all__ = [
    "Node",
    "dimension",
    "define",
    "build_graph",
]




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
        "_reduce_dims",
        "_items",
        "__weakref__",
        "_order",
        "_script_lines",
        "_script",
    )

    _hash: int

    def __init__(
        self,
        fn,
        inputs: dict[str, Any],
        *,
        cache: bool = True,
        dims: tuple[str, ...] = (),
        coords: dict[str, list[Any]] | None = None,
        reduce_dims: Sequence[str] | str = (),
    ):
        """Initialize a Node.

        Parameters
        ----------
        fn : Callable
            The function to execute.
        inputs : dict[str, Any]
            Input arguments for the function. Values can be constant values or other Nodes.
        cache : bool, optional
            Whether to cache the result of this node. Defaults to True.
        dims : tuple[str, ...], optional
            Dimensions associated with this node (if it's a VectorNode).
        coords : dict[str, list[Any]], optional
            Coordinate values for the dimensions.
        reduce_dims : Sequence[str] | str, optional
             Dimensions to reduce (aggregate). Use "all" to reduce all dimensions.
             - If [], no reduction (pure Map/broadcast).
             - If ["time"], reduce "time" dimension.
             - If "all", reduce all input dimensions to scalar.
        """
        self.fn = fn
        self.inputs = inputs
        self.cache = cache
        
        self.dims = dims
        self.coords = coords or {}
        self._reduce_dims = reduce_dims
        self._items = None

        # UNIFIED BROADCAST / REDUCTION LOGIC
        # Identify vector inputs
        vector_inputs = {k: v for k, v in inputs.items() if isinstance(v, Node) and v.dims}
        
        if reduce_dims and vector_inputs:
            # Collect all input dimensions
            all_input_dims = {d for v in vector_inputs.values() for d in v.dims}
            
            # Handle "all" shortcut
            if reduce_dims == "all":
                reduce_set = all_input_dims
            else:
                reduce_set = set(reduce_dims)
                # Validate: reduce_dims must be subset of input dims
                invalid = reduce_set - all_input_dims
                if invalid:
                    from .exceptions import DimensionMismatchError
                    raise DimensionMismatchError(f"reduce_dims {invalid} not in input dims {all_input_dims}")
            
            # Compute output dims (input - reduce)
            output_dims = tuple(d for d in sorted(all_input_dims) if d not in reduce_set)
            
            # Use output_dims as target for broadcasting
            self.dims, self.coords, self._items = _broadcast(fn, inputs, vector_inputs, cache, target_dims=output_dims)
            
            # If the result is a Single Node (Scalar), adopt its wired inputs
            if not self.dims:
                self.inputs = self._items.item().inputs

        elif not dims and vector_inputs:
            # Default: Automatic Broadcasting (Map) over ALL dims
            self.dims, self.coords, self._items = _broadcast(fn, inputs, vector_inputs, cache, target_dims=None)
            
        else:
             # Scalar or Explicit Vector Creation (via decorator)
             pass

        # Logical dependencies for script generation (exclude internal _items)
        self.deps_nodes: list[Node] = list(_collect_nodes(self.inputs.values()))
        
        # Execution dependencies include internal items for runtime scheduling
        self._exec_deps: list[Node] = list(self.deps_nodes)
        if self._items is not None:
            self._exec_deps.extend(self._items.flat)

        # Compute hash
        ignore = frozenset(getattr(fn, "_node_ignore", ()))
        self._hash, _ = compute_node_identity(fn.__name__, inputs, ignore)

    def __getstate__(self):
        """Used for serialization."""
        exclude = {"__weakref__", "_order", "_script", "_script_lines"}
        state = {}
        for k in self.__slots__:
            if k in exclude:
                continue
            try:
                state[k] = getattr(self, k)
            except AttributeError:
                pass
        return state

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    def __repr__(self):
        return self.script

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return False
        return self._hash == other._hash

    def __hash__(self) -> int:
        return self._hash

    @property
    def script_lines(self) -> list[tuple[int, str]]:
        """Generate the script lines for this node's execution graph.
        
        Returns
        -------
        list[tuple[int, str]]
             A list of (hash, source_line) tuples representing the topological execution order.
        """
        try:
            return self._script_lines
        except AttributeError:
            pass

        # Build logical order using deps_nodes (not _exec_deps)
        # This excludes internal _items, showing only logical dependencies
        from graphlib import TopologicalSorter
        from collections import defaultdict
        
        edges: dict[Node, list[Node]] = {}
        stack = [self]
        seen: set[int] = set()
        while stack:
            node = stack.pop()
            if node._hash in seen:
                continue
            seen.add(node._hash)
            deps = node.deps_nodes  # Logical deps only
            edges[node] = deps
            stack.extend(deps)
        
        logical_order = list(TopologicalSorter(edges).static_order())
        
        fn_counters: dict[str, int] = defaultdict(int)
        var_names: dict[Node, str] = {}
        
        for node in logical_order:
            fn_name = node.fn.__name__
            idx = fn_counters[fn_name]
            fn_counters[fn_name] += 1
            var_names[node] = f"{fn_name}_{idx}"
        
        lines: list[tuple[int, str]] = []
        for node in logical_order:
            var_map = {d: var_names[d] for d in node.deps_nodes}
            ignore = getattr(node.fn, "_node_ignore", ())
            call = _render_call(
                node.fn,
                node.inputs,
                mapping=var_map,
                ignore=ignore,
            )
            # Add dimension comment for VectorNodes
            if node.dims:
                dims_comment = ", ".join(f"{d}:{len(node.coords.get(d, []))}" for d in node.dims)
                line = f"{var_names[node]} = {call}  # dims=({dims_comment})"
            else:
                line = f"{var_names[node]} = {call}"
            lines.append((node._hash, line))
        
        self._script_lines = lines
        return lines

    @property
    def order(self) -> list["Node"]:
        """Return the topological execution order of the graph rooted at this node."""
        try:
            return self._order
        except AttributeError:
            self._order, _ = build_graph(self, None)
            return self._order

    @property
    def script(self) -> str:
        """Return a standalone executable script for this node."""
        try:
            return self._script
        except AttributeError:
            pass

        body = "\n".join(line for _, line in self.script_lines)
        self._script = f"# hash = {self._hash:x}\n{body}"
        return self._script

    def __call__(self, *, force: bool = False):
        """Execute this node via the global Runtime.
        
        Parameters
        ----------
        force : bool, optional
            If True, invalidates cache before running.
        
        Returns
        -------
        Any
            The result of the execution.
        """
        from .runtime import get_runtime
        if force:
            self.invalidate()
        return get_runtime().run(self, cache_root=self.cache)

    def invalidate(self) -> None:
        from .runtime import get_runtime
        get_runtime().delete(self)

def _collect_nodes(v):
    from collections.abc import Iterator
    if isinstance(v, Iterator) and not isinstance(v, (str, bytes)):
        raise TypeError(f"Node inputs cannot be Iterator/Generator (type: {type(v).__name__}). Materialize them as list or tuple first.")
    if isinstance(v, Node):
        yield v
    elif isinstance(v, Mapping):
        for item in v.values():
            yield from _collect_nodes(item)
    elif isinstance(v, Iterable) and not isinstance(v, (str, bytes)):
        # Numpy 0-d arrays are Iterable but raise TypeError on iter()
        if getattr(v, "ndim", 1) == 0:
            return
        for item in v:
            yield from _collect_nodes(item)

# Late imports
from .dimension import dimension, broadcast as _broadcast
from .hashing import compute_node_identity, _canonical


def build_graph(
    root: "Node", cache: "Cache | None"
) -> tuple[list["Node"], dict["Node", list["Node"]]]:
    """Return topological order and dependency graph.

    When ``cache`` is provided, traversal stops at nodes with a cache hit,
    effectively pruning their ancestors from the resulting graph.
    """
    
    edges: dict["Node", list["Node"]] = {}
    stack = [root]
    seen: set["Node"] = set()
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        hit = cache is not None and node.cache and cache.get(node.fn.__name__, node._hash)[0]
        if hit:
            edges[node] = []
            continue
        deps = node._exec_deps
        edges[node] = deps
        stack.extend(deps)
    order = list(TopologicalSorter(edges).static_order())
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
        Whether to cache the result. Defaults to True.
    local : bool, optional
        Execute directly in the caller thread, bypassing any executor. 
        Defaults to False.
    reduce_dims : Sequence[str] | str, optional
        Dimensions to reduce over when aggregating results.

    Returns
    -------
    Callable
        A decorator that converts the function into a Node factory.

    Examples
    --------
    >>> import time
    >>> @node.define(workers=2)
    ... def slow_task(x):
    ...     time.sleep(1)
    ...     return x
    """
    ignore_set = set(ignore or [])

    def deco(fn: Callable[..., Any]) -> Callable[..., Node]:
        # Check for closure variables
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
            "_node_local": local,
        }

        for k, v in node_attrs.items():
            setattr(fn, k, v)

        validated_fn: Callable[..., Any] | None = None

        def get_validated_fn() -> Callable[..., Any]:
            nonlocal validated_fn
            if validated_fn is None:
                validated_fn = validate_call(
                    fn,
                    config={"arbitrary_types_allowed": True},
                )
                for k, v in node_attrs.items():
                    setattr(validated_fn, k, v)
            return validated_fn

        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Node:
            # Runtime is only required here, at node instantiation time
            from .runtime import get_runtime
            
            current_runtime = get_runtime()
            effective_workers = (
                workers if workers is not None else current_runtime.workers
            )
            if current_runtime.validate:
                selected_fn = get_validated_fn()
            else:
                selected_fn = fn
            setattr(selected_fn, "_node_workers", effective_workers)
            bound = sig_obj.bind_partial(*args, **kwargs)
            for name, val in current_runtime.config.defaults(
                fn.__name__,
                runtime=current_runtime,
            ).items():
                if name not in bound.arguments:
                    bound.arguments[name] = val
            bound.apply_defaults()

            node = Node(selected_fn, bound.arguments, cache=cache, reduce_dims=reduce_dims)
            cached_node = current_runtime._registry.get(node)
            if cached_node is not None:
                return cached_node
            current_runtime._registry[node] = node
            return node

        wrapper.__signature__ = sig_obj  # type: ignore[attr-defined]
        return wrapper

    return deco
