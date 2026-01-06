"""Core Node definition and DAG graph utilities."""

from __future__ import annotations

import functools
import hashlib
import inspect
import threading
import warnings
from collections.abc import Iterable, Mapping, Sequence
from graphlib import TopologicalSorter
from typing import Any, Callable, Dict, List, Tuple, cast, TYPE_CHECKING

from cachetools import LRUCache

if TYPE_CHECKING:
    from .cache import Cache

__all__ = [
    "Node",
    "gather",
    "map",
]

# Global caches & locks for canonical/render
_can_lock = threading.Lock()
_ren_lock = threading.Lock()
_canonical_cache: LRUCache[tuple[int, str], str] = LRUCache(maxsize=4096)
_render_cache: LRUCache[tuple, str] = LRUCache(maxsize=2048)


def _is_safe_type(obj: Any, depth: int = 0) -> tuple[bool, str]:
    """检查对象是否有稳定的规范化表示。
    
    Returns:
        (is_safe, reason): 是否安全及原因
    """
    # 防止无限递归
    if depth > 10:
        return False, "nested too deeply"
    
    # 基本类型
    if isinstance(obj, (int, float, str, bool, type(None))):
        return True, ""
    
    # Node 类型（延迟导入时使用字符串检查）
    if type(obj).__name__ == "Node":
        return True, ""
    
    # 列表/元组
    if isinstance(obj, (list, tuple)):
        for item in obj:
            safe, reason = _is_safe_type(item, depth + 1)
            if not safe:
                return False, f"contains unsafe element: {reason}"
        return True, ""
    
    # 字典
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, (int, float, str, bool, type(None))):
                return False, f"dict key {k!r} is not a simple type"
            safe, reason = _is_safe_type(v, depth + 1)
            if not safe:
                return False, f"dict value: {reason}"
        return True, ""
    
    # 集合
    if isinstance(obj, set):
        for item in obj:
            safe, reason = _is_safe_type(item, depth + 1)
            if not safe:
                return False, f"set element: {reason}"
        return True, ""
    
    # 其他类型：尝试验证 repr 稳定性
    try:
        r1 = repr(obj)
        r2 = repr(obj)
        if r1 != r2:
            return False, f"unstable repr for {type(obj).__name__}"
        # 检测可能的内存地址
        if "at 0x" in r1 or ("<" in r1 and ">" in r1 and "object" in r1.lower()):
            return False, f"repr likely contains memory address for {type(obj).__name__}"
        return True, ""
    except Exception as e:
        return False, f"repr failed: {e}"


def _canonical(obj: Any) -> str:
    """Convert an object into a deterministic string representation."""
    if isinstance(obj, Node):
        return obj.key

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


def compute_hash(fn_name: str, inputs: Dict[str, Any], ignore: Set[str] = frozenset()) -> int:
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
    mapping: Dict["Node", str] | None = None,
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
    mapping : Dict[Node, str]
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
        res = _render_cache.get(key)
    if res is not None:
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
        hit = False
        if cache is not None and node.cache:
            hit, _ = cache.get(node.key)
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
    """

    __slots__ = (
        "fn",
        "inputs",
        "deps_nodes",
        "cache",
        "_hash",
        "_lock",
        "_runtime",
        "__dict__",
        "__weakref__",
    )

    _hash: int
    _lock: threading.Lock

    def __init__(
        self,
        fn,
        inputs: Dict[str, Any],
        *,
        cache: bool = True,
        runtime: Any = None,
    ):
        """初始化 Node。

        Parameters
        ----------
        fn : Callable
            被包装的函数
        inputs : Dict[str, Any]
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
        
        self.deps_nodes: List[Node] = list(
            node for value in inputs.values() for node in collect_nodes(value)
        )

        # TODO: 循环检测暂时禁用
        # if any(d is self for d in self.deps_nodes):
        #     raise ValueError("Cycle detected in DAG")

        # 计算 hash（规范化逻辑封装在 compute_hash 中）
        ignore = set(getattr(fn, "_node_ignore", ()))
        self._hash = compute_hash(fn.__name__, inputs, ignore)
        self._lock = threading.Lock()

    def __getstate__(self):
        return {k: getattr(self, k) for k in self.__slots__ if k not in ("_lock", "_runtime")}

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

    @functools.cached_property
    def script_lines(self) -> List[Tuple[int, str]]:
        """Return script lines for this node with simplified variable names."""
        with self._lock:
            order = self.order
            # 为每个函数维护计数器，生成简化变量名
            fn_counters: Dict[str, int] = {}
            var_names: Dict[Node, str] = {}
            
            for node in order:
                fn_name = node.fn.__name__
                idx = fn_counters.get(fn_name, 0)
                fn_counters[fn_name] = idx + 1
                var_names[node] = f"{fn_name}_{idx}"
            
            lines: List[Tuple[int, str]] = []
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
            return lines

    @functools.cached_property
    def order(self) -> List["Node"]:
        order, _ = _build_graph(self, None)
        return order

    @functools.cached_property
    def script(self) -> str:
        """Return human-readable script with hash header."""
        body = "\n".join(line for _, line in self.script_lines)
        return f"# hash = {self._hash:x}\n{body}"

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

    ``nodes`` may be passed either as positional arguments or as a single
    iterable.  The returned node produces a list of each input node's value
    in the provided order.  ``workers`` controls the concurrent executions
    of the gather node itself.
    """
    from .runtime import get_runtime

    if len(nodes) == 1 and not isinstance(nodes[0], Node):
        nodes_list = tuple(cast(Iterable[Node], nodes[0]))
    else:
        nodes_list = cast(Tuple[Node, ...], nodes)

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


def map(
    node_fn: Callable[..., Node],
    *,
    workers: int | None = None,
    cache: bool = True,
    **iterables: Iterable[Any],
) -> Node:
    """Map a node function over parameter iterables.

    Each keyword argument value is an iterable to map over. When multiple
    iterables are provided, they are zipped together. The result is a
    ``gather`` node containing all the mapped nodes.

    Parameters
    ----------
    node_fn : Callable
        A ``@node.define`` decorated function.
    workers : int, optional
        Concurrency for the resulting gather node.
    cache : bool
        Whether to cache the gather result (individual nodes still cache).
    **iterables
        Keyword arguments where each value is an iterable. The keys are
        parameter names to bind.

    Returns
    -------
    Node
        A gather node containing all mapped nodes.

    Examples
    --------
    >>> # Single parameter
    >>> node.map(process, x=[1, 2, 3])
    >>> # Equivalent to: gather([process(x=1), process(x=2), process(x=3)])

    >>> # Multiple parameters (zipped)
    >>> node.map(compare, a=[1, 2], b=[3, 4])
    >>> # Equivalent to: gather([compare(a=1, b=3), compare(a=2, b=4)])

    >>> # Pass lists as parameter values
    >>> node.map(process_batch, items=[[1, 2], [3, 4]])
    >>> # Creates: process_batch(items=[1, 2]), process_batch(items=[3, 4])
    """
    if not iterables:
        raise ValueError("map() requires at least one iterable keyword argument")

    keys = list(iterables.keys())
    values = [list(v) for v in iterables.values()]

    # Check all iterables have the same length
    lengths = [len(v) for v in values]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"All iterables must have the same length, got {dict(zip(keys, lengths))}"
        )

    if not values[0]:
        # Empty iterables - return empty gather
        # Create a dummy node just to get the runtime
        from .runtime import get_runtime
        runtime = get_runtime()

        @runtime.define(workers=workers, cache=cache)
        def _empty_gather():
            return []

        return _empty_gather()

    # Build nodes by zipping iterables
    nodes = [
        node_fn(**dict(zip(keys, args)))
        for args in zip(*values)
    ]

    return gather(nodes, workers=workers, cache=cache)

