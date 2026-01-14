"""Core Node definition and DAG graph utilities."""

from __future__ import annotations


import hashlib
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from graphlib import TopologicalSorter
from typing import Any, TYPE_CHECKING, cast

if TYPE_CHECKING:
    from .cache import Cache

__all__ = [
    "Node",
    "gather",
    "sweep",
]


def _is_safe_type(obj: Any, depth: int = 0) -> tuple[bool, str]:
    """Check if an object has a stable canonical representation.
    
    Only checks two dangerous cases:
    1. Nested too deeply (may cause infinite recursion)
    2. Dict keys are not simple types
    
    Args:
        obj: The object to check.
        depth: Current recursion depth.
        
    Returns:
        A tuple of (is_safe, reason).
    """
    if depth > 10:
        return False, "nested too deeply"
    
    # Recursively check container types
    if isinstance(obj, (list, tuple)):
        for item in obj:
            safe, reason = _is_safe_type(item, depth + 1)
            if not safe:
                return False, f"contains unsafe element: {reason}"
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, (int, float, str, bool, type(None))):
                return False, f"dict key {k!r} is not a simple type"
            safe, reason = _is_safe_type(v, depth + 1)
            if not safe:
                return False, f"dict value: {reason}"
    elif isinstance(obj, set):
        for item in obj:
            safe, reason = _is_safe_type(item, depth + 1)
            if not safe:
                return False, f"set element: {reason}"
    
    return True, ""


def _canonical(obj: Any) -> str:
    """Convert an object into a deterministic string representation."""
    if isinstance(obj, Node):
        return f"{obj.fn.__name__}_{obj._hash:x}"
    
    # Functions decorated with @node.define: use function name as canonical form
    if callable(obj) and hasattr(obj, "_node_sig"):
        return f"NodeFactory:{obj.__name__}"
    
    # Regular functions/methods: use __qualname__ to avoid memory address
    if callable(obj) and hasattr(obj, "__qualname__"):
        return f"Func:{obj.__qualname__}"

    # Type safety check
    safe, reason = _is_safe_type(obj)
    if not safe:
        warnings.warn(
            f"Parameter of type {type(obj).__name__} may not have stable "
            f"canonical representation: {reason}. "
            f"This could lead to unexpected cache behavior.",
            category=UserWarning,
            stacklevel=4,
        )

    if isinstance(obj, dict):
        inner = ", ".join(f"{repr(k)}: {_canonical(v)}" for k, v in sorted(obj.items()))
        return "{" + inner + "}"
    elif isinstance(obj, (list, tuple)):
        inner = ", ".join(_canonical(v) for v in obj)
        return "[" + inner + "]" if isinstance(obj, list) else "(" + inner + ")"
    elif isinstance(obj, set):
        inner = ", ".join(_canonical(v) for v in sorted(obj))
        return "{" + inner + "}"
    else:
        return repr(obj)


def compute_node_identity(
    fn_name: str, 
    inputs: dict[str, Any], 
    ignore: frozenset[str] = frozenset()
) -> tuple[int, tuple[tuple[str, str], ...]]:
    """Compute unique node identity hash and canonical inputs.
    
    Normalizes parameters internally:
    - Node parameters use their hash
    - Other parameters use _canonical for deterministic string representation
    - Sorted by parameter name for cross-run determinism
    
    Args:
        fn_name: Function name.
        inputs: Mapping of parameter names to values.
        ignore: Parameter names to exclude from hash computation.
        
    Returns:
        A tuple of (64-bit hash, canonical inputs tuple).
    """
    canonical_inputs = tuple(
        sorted(
            (k, f"{hash(v):016x}" if isinstance(v, Node) else _canonical(v))
            for k, v in inputs.items()
            if k not in ignore
        )
    )
    hash_source = (fn_name, canonical_inputs)
    hash_value = int(
        hashlib.blake2b(repr(hash_source).encode(), digest_size=8).hexdigest(),
        16
    )
    return hash_value, canonical_inputs


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
            if mapping:
                return mapping[v]
            return _render_call(
                v.fn,
                v.inputs,
                ignore=getattr(v.fn, "_node_ignore", ()),
            )
        # Functions decorated with @node.define: render as function name
        if callable(v) and hasattr(v, "_node_sig"):
            return v.__name__
        if isinstance(v, (list, tuple)):
            inner = ", ".join(render(item) for item in v)
            return f"[{inner}]" if isinstance(v, list) else f"({inner})"
        return _canonical(v)

    ignore_set = set(ignore or ())
    parts = [f"{k}={render(v)}" for k, v in inputs.items() if k not in ignore_set]
    return f"{fn.__name__}({', '.join(parts)})"


def _build_graph(
    root: "Node", cache: "Cache | None"
) -> tuple[list["Node"], dict["Node", list["Node"]]]:
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
        hit = cache is not None and node.cache and cache.get(node.fn.__name__, node._hash)[0]
        if hit:
            edges[node] = []
            continue
        deps = sorted(node.deps_nodes)
        edges[node] = deps
        stack.extend(deps)
    order = list(TopologicalSorter(edges).static_order())
    return order, edges


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
        "_canonical_inputs",
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
    ):
        """Initialize a Node.

        Args:
            fn: The wrapped function.
            inputs: Mapping of parameter names to values (includes all defaults).
            cache: Whether to enable caching for this node.
        """
        self.fn = fn
        self.inputs = inputs
        self.cache = cache

        # Recursively collect all Node objects from parameters (including nested in tuple/list)
        def collect_nodes(v):
            if isinstance(v, Node):
                yield v
            elif isinstance(v, (tuple, list)):
                for item in v:
                    yield from collect_nodes(item)
        
        self.deps_nodes: list[Node] = list(
            node for value in inputs.values() for node in collect_nodes(value)
        )
        # Compute hash and canonical inputs
        ignore = frozenset(getattr(fn, "_node_ignore", ()))
        self._hash, self._canonical_inputs = compute_node_identity(fn.__name__, inputs, ignore)

    def __getstate__(self):
        # Exclude non-serializable attributes and lazy attributes that may not exist
        exclude = {"__weakref__"}
        return {k: getattr(self, k) for k in self.__slots__ if k not in exclude and hasattr(self, k)}

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


    def __lt__(self, other: "Node") -> bool:
        """Compare nodes by function name, then by canonical inputs."""
        if self.fn.__name__ != other.fn.__name__:
            return self.fn.__name__ < other.fn.__name__
        return self._canonical_inputs < other._canonical_inputs


    @property
    def script_lines(self) -> list[tuple[int, str]]:
        """Return script lines for this node with simplified variable names."""
        if hasattr(self, '_script_lines'):
            return self._script_lines


        order = self.order
        # Maintain counter per function name to generate simplified variable names
        from collections import defaultdict
        fn_counters: dict[str, int] = defaultdict(int)
        var_names: dict[Node, str] = {}
        
        for node in order:
            fn_name = node.fn.__name__
            idx = fn_counters[fn_name]
            fn_counters[fn_name] += 1
            var_names[node] = f"{fn_name}_{idx}"
        
        lines: list[tuple[int, str]] = []
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
            self._order, _ = _build_graph(self, None)
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

    # Aliases for backward compatibility (deprecated)
    def get(self):
        """Execute this node using the runtime.
        
        .. deprecated::
            Use ``node()`` instead.
        """
        return self()

    def delete(self) -> None:
        """Delete cached value for this node.
        
        .. deprecated::
            Use ``node.invalidate()`` instead.
        """
        self.invalidate()

    def create(self):
        """Recompute this node ignoring existing cache.
        
        .. deprecated::
            Use ``node(force=True)`` instead.
        """
        return self(force=True)


def gather(
    *nodes: Node | Iterable[Node],
    workers: int | None = None,
    cache: bool = True,
) -> Node:
    """Aggregate multiple nodes into a single list result.

    This function creates a new Node that, when executed, will run all input
    nodes (potentially in parallel) and return a list of their results in the
    same order.

    Parameters
    ----------
    *nodes : Node | Iterable[Node]
        Nodes to gather. Can be passed as positional arguments or as a single
        iterable (list, tuple, generator).
    workers : int, optional
        Max concurrency for executing these nodes.
    cache : bool, optional
        Whether to cache the gathered result itself. Defaults to True.

    Returns
    -------
    Node
        A node that evaluates to ``List[Any]``.

    Examples
    --------
    >>> tasks = [my_task(i) for i in range(5)]
    >>> all_results = node.gather(tasks)()
    """
    from .runtime import get_runtime

    if len(nodes) == 1 and not isinstance(nodes[0], Node):
        nodes_list = tuple(cast(Iterable[Node], nodes[0]))
    else:
        nodes_list = cast(tuple[Node, ...], nodes)

    if not nodes_list:
        raise ValueError("no nodes provided")

    runtime = get_runtime()

    @runtime.define(workers=workers, cache=cache)
    def _gather(*items):
        return list(items)

    return _gather(*nodes_list)


def sweep(
    target: Callable[..., "Node"],
    config: dict[str, Iterable[Any]],
    *,
    workers: int | None = None,
    cache: bool = True,
    **kwargs: Any,
) -> "Node":
    """Scan a node over different configuration values.

    For each combination of config values, sets the global configuration,
    then calls the target function to create a new Node. The config changes
    are reflected in parameter defaults through OmegaConf references.

    Parameters
    ----------
    target : Callable[..., Node]
        The node-decorated function to call (e.g., ``my_func``, not ``my_func()``).
    config : dict[str, Iterable[Any]]
        Mapping of config keys to values to scan over. Keys are dot-separated
        paths into the global config (e.g., "multiplier" sets ``node.cfg.multiplier``).
    workers : int, optional
        Concurrency limit for the sweep execution.
    cache : bool, optional
        Whether to cache the result list. Defaults to True.
    **kwargs
        Additional keyword arguments to pass to the target function.

    Returns
    -------
    Node
        A node that evaluates to a list of results, one for each config value.

    Examples
    --------
    >>> # Config: base_value.multiplier: ${global_multiplier}
    >>> @node.define()
    ... def base_value(x: int, multiplier: int):
    ...     return x * multiplier
    >>>
    >>> # Sweep over different multiplier values
    >>> results = node.sweep(
    ...     base_value,  # Pass the function, not base_value()
    ...     config={"global_multiplier": [1, 2, 3]},
    ...     x=5,  # Pass required args as kwargs
    ... )()
    >>> # [5, 10, 15]
    """
    from .runtime import get_runtime

    # Convert iterables to lists, validate, and collect lengths in one pass
    config_lists = {}
    lengths = {}
    for k, v in config.items():
        items = list(v)
        if not items:
            raise ValueError(f"Config key '{k}' has empty iterable")
        config_lists[k] = items
        lengths[k] = len(items)

    # Check all config values have the same length
    if len(set(lengths.values())) > 1:
        raise ValueError(
            f"All config iterables must have the same length, got {lengths}"
        )

    n_items = next(iter(lengths.values()))
    runtime = get_runtime()
    cfg = runtime.config._conf

    # Helper to set config value
    def set_config(key: str, value: Any):
        parts = key.split(".")
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    # Create a node for each config value
    nodes = []
    for i in range(n_items):
        # Set config values
        for key, values in config_lists.items():
            set_config(key, values[i])
        # Call the target function to create a new Node
        # The Node's hash will include the current config values via defaults
        new_node = target(**kwargs)
        nodes.append(new_node)

    # Use gather to collect all results
    return gather(*nodes, workers=workers, cache=cache)
