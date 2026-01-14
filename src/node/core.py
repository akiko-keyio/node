"""Core Node definition and DAG graph utilities."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .cache import Cache

__all__ = [
    "Node",
    "gather",
    "dimension",
]




def _render_call(
    fn: Callable,
    inputs: Mapping[str, Any],
    *,
    mapping: dict["Node", str] | None = None,
    ignore: Sequence[str] | None = None,
) -> str:
    """Render a function call as a string.

    Args:
        fn: The function object.
        inputs: Mapping of parameter names to values.
        mapping: Mapping of Node to variable names.
        ignore: Parameter names to exclude from rendering.
        
    Returns:
        A string representation of the function call.
    """
    def render(v: Any) -> str:
        if isinstance(v, Node):
            # Use variable name if already assigned in this script context
            if mapping and v in mapping:
                return mapping[v]
            
            # Recursive rendering for nested/unmapped nodes
            return _render_call(
                v.fn,
                v.inputs,
                mapping=mapping, 
                ignore=getattr(v.fn, "_node_ignore", ()),
            )
            
        # Custom node definitions: use function name
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
    """Represents a computation unit in the DAG.

    A Node is a pure data structure that describes a function call with its
    arguments. It does not execute itself; execution is handled by the Runtime.

    Attributes
    ----------
    fn : Callable
        The function being wrapped by this node.
    inputs : Dict[str, Any]
        Mapping of argument names to their values.
    deps_nodes : List[Node]
        List of other Node objects that this node depends on.
    cache : bool
        Whether caching is enabled for this node.
    _hash : int
        Unique identifier based on function name and arguments.

    Examples
    --------
    >>> @node.define()
    ... def double(x):
    ...     return x * 2
    >>>
    >>> n = double(5)  # Create a node
    >>> n()            # Execute and get result
    10
    >>> n.script       # Get execution script
    '# hash = ...\\nv0 = double(x=5)'
    """
    __slots__ = (
        "fn",
        "inputs",
        "deps_nodes",
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
        fn,
        inputs: dict[str, Any],
        *,
        cache: bool = True,
        dims: tuple[str, ...] = (),
        coords: dict[str, list[Any]] | None = None,
    ):
        self.fn = fn
        self.inputs = inputs
        self.cache = cache
        
        self.dims = dims
        self.coords = coords or {}
        self._items = None

        if not dims:
            vector_inputs = {k: v for k, v in inputs.items() if isinstance(v, Node) and v.dims}
            if vector_inputs:
                self.dims, self.coords, self._items = _broadcast(fn, inputs, vector_inputs, cache)

        # Recursively collect all Node objects from parameters
        self.deps_nodes: list[Node] = list(_collect_nodes(inputs.values()))

        # Compute hash
        ignore = frozenset(getattr(fn, "_node_ignore", ()))
        self._hash, _ = compute_node_identity(fn.__name__, inputs, ignore)

    def __getstate__(self):
        """Used for serialization (e.g. pickling)."""
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
        # Logical View (Script) is primary
        s = self.script
        # Append dimensions info as a comment for Vector Nodes
        if self.dims:
            dims_str = ", ".join(f"{d}:{len(c)}" for d, c in self.coords.items())
            return f"{s} # <VectorNode dims=({dims_str})>"
        return s

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return False
        return self._hash == other._hash

    def __hash__(self) -> int:
        return self._hash

    @property
    def script_lines(self) -> list[tuple[int, str]]:
        """Return script lines for this node with simplified variable names."""
        try:
            return self._script_lines
        except AttributeError:
            pass

        order = self.order
        # Maintain counter per function name to generate simplified variable names
        from collections import defaultdict
        fn_counters: dict[str, int] = defaultdict(int)
        var_names: dict[Node, str] = {}
        
        # Pass 1: Assign variable names
        for node in order:
            fn_name = node.fn.__name__
            idx = fn_counters[fn_name]
            fn_counters[fn_name] += 1
            var_names[node] = f"{fn_name}_{idx}"
        
        lines: list[tuple[int, str]] = []
        # Pass 2: Generate lines
        for node in order:
            var_map = {d: var_names[d] for d in node.deps_nodes}
            ignore = getattr(node.fn, "_node_ignore", ())
            
            call = _render_call(
                node.fn,
                node.inputs,
                mapping=var_map,
                ignore=ignore,
            )
            lines.append((node._hash, f"{var_names[node]} = {call}"))
        
        self._script_lines = lines
        return lines

    @property
    def order(self) -> list["Node"]:
        try:
            return self._order
        except AttributeError:
            self._order, _ = build_graph(self, None)
            return self._order

    @property
    def script(self) -> str:
        """Return human-readable script with hash header."""
        try:
            return self._script
        except AttributeError:
            pass

        body = "\n".join(line for _, line in self.script_lines)
        self._script = f"# hash = {self._hash:x}\n{body}"
        return self._script

    # Primary API
    def __call__(self, *, force: bool = False):
        """Execute this node and return the result.
        
        Parameters
        ----------
        force : bool, optional
            If True, invalidate cache for this node and recompute. 
            Defaults to False.
            
        Returns
        -------
        Any
            The computed result.
            
        Examples
        --------
        >>> result = my_node()            # Execute with caching
        >>> result = my_node(force=True)  # Force recomputation
        """
        from .runtime import get_runtime
        if force:
            self.invalidate()
        return get_runtime().run(self, cache_root=self.cache)

    def invalidate(self) -> None:
        """Invalidate cached value for this node.
        
        After calling this, the next execution will recompute the result.
        """
        from .runtime import get_runtime
        get_runtime().delete(self)

# Helper utilities
def _collect_nodes(v):
    """Recursively yield all Node objects from containers."""
    if isinstance(v, Node):
        yield v
    elif isinstance(v, Mapping):
        for item in v.values():
            yield from _collect_nodes(item)
    elif isinstance(v, Iterable) and not isinstance(v, (str, bytes)):
        for item in v:
            yield from _collect_nodes(item)

# Late imports to avoid circular dependencies
from .dimension import dimension, broadcast as _broadcast
from .identity import compute_node_identity, _canonical
from .ops import gather
from .graph import build_graph

