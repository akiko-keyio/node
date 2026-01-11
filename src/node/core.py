"""Core Node definition and DAG graph utilities."""

from __future__ import annotations


import hashlib
import inspect
import threading
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from graphlib import TopologicalSorter
from typing import Any, TYPE_CHECKING, cast

from cachetools import LRUCache

if TYPE_CHECKING:
    from .cache import Cache

__all__ = [
    "Node",
    "gather",
    "sweep",
]

# Global caches & locks for canonical/render
_can_lock = threading.Lock()
_ren_lock = threading.Lock()
_canonical_cache: LRUCache[tuple[int, str], str] = LRUCache(maxsize=4096)
_render_cache: LRUCache[tuple, str] = LRUCache(maxsize=2048)

def _is_safe_type(obj: Any, depth: int = 0) -> tuple[bool, str]:
    """检查对象是否有稳定的规范化表示。
    
    仅检查两种危险情况：
    1. 嵌套过深（可能导致无限递归）
    2. 字典键不是简单类型
    
    Returns:
        (is_safe, reason): 是否安全及原因
    """
    if depth > 10:
        return False, "nested too deeply"
    
    # 递归检查容器类型
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
        return obj.key
    
    # @node.define 装饰过的函数：用函数名作为规范表示
    if callable(obj) and hasattr(obj, "_node_sig"):
        return f"NodeFactory:{obj.__name__}"
    
    # 普通函数/方法：用 __qualname__ 避免内存地址
    if callable(obj) and hasattr(obj, "__qualname__"):
        return f"Func:{obj.__qualname__}"

    # 类型安全检查
    safe, reason = _is_safe_type(obj)
    if not safe:
        warnings.warn(
            f"Parameter of type {type(obj).__name__} may not have stable "
            f"canonical representation: {reason}. "
            f"This could lead to unexpected cache behavior.",
            category=UserWarning,
            stacklevel=4,
        )

    # 使用 (id, type_name) 作为缓存键，避免对象销毁后 id 被重用
    cache_key = (id(obj), type(obj).__name__)
    with _can_lock:
        if (res := _canonical_cache.get(cache_key)) is not None:
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
        _canonical_cache[cache_key] = res
    return res


def compute_hash(fn_name: str, inputs: dict[str, Any], ignore: frozenset[str] = frozenset()) -> int:
    """
    计算节点的唯一标识哈希。
    
    内部对参数进行规范化：
    - Node 参数用其 hash 表示
    - 其他参数用 _canonical 转为确定性字符串
    - 按参数名排序保证跨运行确定性
    
    Parameters
    ----------
    fn_name : str
        函数名
    inputs : Dict[str, Any]
        参数名到值的映射
    ignore : Set[str]
        需要忽略的参数名（不参与 hash 计算）
        
    Returns
    -------
    int
        12 位 hex 对应的整数 hash
    """
    canonical_inputs = tuple(
        sorted(
            (k, f"{hash(v):012x}" if isinstance(v, Node) else _canonical(v))
            for k, v in inputs.items()
            if k not in ignore
        )
    )
    hash_source = (fn_name, canonical_inputs)
    return int(
        hashlib.blake2b(repr(hash_source).encode(), digest_size=6).hexdigest(),
        16
    )


def _render_call(
    fn: Callable,
    inputs: Mapping[str, Any],
    *,
    canonical: bool = False,
    mapping: dict["Node", str] | None = None,
    ignore: Sequence[str] | None = None,
) -> str:
    """渲染函数调用为字符串。

    Parameters
    ----------
    fn : Callable
        函数对象
    inputs : Mapping[str, Any]
        参数名到值的映射
    canonical : bool
        是否使用规范化表示
    mapping : dict[Node, str]
        Node 到变量名的映射
    ignore : Sequence[str]
        需要忽略的参数名
    """

    def render(v: Any) -> str:
        if isinstance(v, Node):
            if mapping:
                return mapping[v]
            return _render_call(
                v.fn,
                v.inputs,
                canonical=canonical,
                ignore=getattr(v.fn, "_node_ignore", ()),
            )
        # @node.define 装饰过的函数：渲染为函数名
        if callable(v) and hasattr(v, "_node_sig"):
            return v.__name__
        return _canonical(v) if canonical else repr(v)

    def key_of(v: Any) -> str:
        if isinstance(v, Node):
            return v.key
        return repr(v)

    # 缓存 key 基于 inputs
    key = (
        fn.__qualname__,
        canonical,
        tuple(sorted((k, key_of(v)) for k, v in inputs.items())),
        tuple(sorted((d.key, v) for d, v in mapping.items())) if mapping else None,
        tuple(sorted(ignore or [])),
    )
    with _ren_lock:
        if (res := _render_cache.get(key)) is not None:
            return res


    ignore_set = set(ignore or ())
    parts = [f"{k}={render(v)}" for k, v in inputs.items() if k not in ignore_set]
    res = f"{fn.__name__}({', '.join(parts)})"

    with _ren_lock:
        _render_cache[key] = res
    return res


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
        hit = cache is not None and node.cache and cache.get(node.key)[0]
        if hit:
            edges[node] = []
            continue
        deps = sorted(node.deps_nodes)
        edges[node] = deps
        stack.extend(deps)
    order = list(TopologicalSorter(edges).static_order())
    return order, edges


def clear_caches() -> None:
    """Clear global helper caches."""
    with _can_lock:
        _canonical_cache.clear()
    with _ren_lock:
        _render_cache.clear()


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
    _lock : threading.Lock
        Lock for thread-safe operations.
    _runtime : Any
        The associated runtime instance.

    Examples
    --------
    >>> @node.define()
    ... def double(x):
    ...     return x * 2
    >>>
    >>> n = double(5)  # Create a node
    >>> n.get()        # Execute and get result
    10
    >>> n.script       # Get execution script
    '# hash = ...\\nv0 = double(x=5)'

    Methods
    -------
    get()
        Execute node and return result.
    delete()
        Delete cached value.
    create()
        Force recompute ignoring cache.
    script()
        Return human-readable script.
    """
    __slots__ = (
        "fn",
        "inputs",
        "deps_nodes",
        "cache",
        "_hash",
        "_lock",
        "_runtime",
        "__weakref__",
        "_order",
        "_script_lines",
        "_script",
    )

    _hash: int
    _lock: threading.Lock

    def __init__(
        self,
        fn,
        inputs: dict[str, Any],
        *,
        cache: bool = True,
        runtime: Any = None,
    ):
        """初始化 Node。

        Parameters
        ----------
        fn : Callable
            被包装的函数
        inputs : dict[str, Any]
            参数名到值的映射（已包含所有默认值）
        cache : bool
            是否启用缓存
        runtime : Runtime
            关联的运行时实例
        """
        self._runtime = runtime
        self.fn = fn
        self.inputs = inputs
        self.cache = cache

        # 递归收集参数中所有的 Node 对象（包括嵌套在 tuple/list 中的）
        def collect_nodes(v):
            if isinstance(v, Node):
                yield v
            elif isinstance(v, (tuple, list)):
                for item in v:
                    yield from collect_nodes(item)
        
        self.deps_nodes: list[Node] = list(
            node for value in inputs.values() for node in collect_nodes(value)
        )
        # 计算 hash（规范化逻辑封装在 compute_hash 中）
        ignore = frozenset(getattr(fn, "_node_ignore", ()))
        self._hash = compute_hash(fn.__name__, inputs, ignore)
        self._lock = threading.Lock()

    def __getstate__(self):
        # Exclude non-serializable attributes and lazy attributes that may not exist
        exclude = {"_lock", "_runtime", "__weakref__"}
        return {k: getattr(self, k) for k in self.__slots__ if k not in exclude and hasattr(self, k)}

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)
        self._lock = threading.Lock()
        self._runtime = None

    def __repr__(self):
        return self.script

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return False
        return self._hash == other._hash

    def __hash__(self) -> int:
        return self._hash

    @property
    def key(self) -> str:
        """Unique identifier combining function name and hash."""
        return f"{self.fn.__name__}_{self._hash:x}"

    def __lt__(self, other: "Node") -> bool:
        return self._hash < other._hash


    @property
    def script_lines(self) -> list[tuple[int, str]]:
        """Return script lines for this node with simplified variable names."""
        if hasattr(self, '_script_lines'):
            return self._script_lines


        with self._lock:
            order = self.order
            # 为每个函数维护计数器，生成简化变量名
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
                    canonical=True,
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

    def _get_runtime(self):
        """Get the runtime for this node, falling back to global."""
        if self._runtime is not None:
            return self._runtime
        from .runtime import get_runtime
        return get_runtime()

    # Convenience methods that delegate to runtime
    def get(self):
        """Execute this node using the runtime."""
        return self._get_runtime().run(self, cache_root=self.cache)

    def delete(self) -> None:
        """Delete cached value for this node."""
        self._get_runtime().delete(self)

    def create(self):
        """Recompute this node ignoring existing cache."""
        return self._get_runtime().create(self)

    def generate(self) -> None:
        """Compute and cache this node without returning the value."""
        self._get_runtime().generate(self)


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
    >>> all_results = node.gather(tasks).get()
    """
    from .runtime import get_runtime

    if len(nodes) == 1 and not isinstance(nodes[0], Node):
        nodes_list = tuple(cast(Iterable[Node], nodes[0]))
    else:
        nodes_list = cast(tuple[Node, ...], nodes)

    if not nodes_list:
        raise ValueError("no nodes provided")

    # Get runtime from first node, or fall back to global
    first_runtime = nodes_list[0]._runtime
    if first_runtime is None:
        runtime = get_runtime()
    else:
        runtime = first_runtime

    # Check all nodes share the same runtime
    for n in nodes_list[1:]:
        if n._runtime is not None and n._runtime is not runtime:
            raise ValueError("nodes belong to different Flow instances")

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
    ... ).get()
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
