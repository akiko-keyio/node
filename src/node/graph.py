"""Graph algorithms and traversal utilities."""
from __future__ import annotations

from graphlib import TopologicalSorter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Node
    from .cache import Cache

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
        deps = node.deps_nodes
        edges[node] = deps
        stack.extend(deps)
    order = list(TopologicalSorter(edges).static_order())
    return order, edges
