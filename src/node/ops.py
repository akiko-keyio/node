"""High-level DAG operations."""
from __future__ import annotations

from collections.abc import Iterable, Callable
from typing import Any, cast, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .core import Node


def gather(
    *nodes: Node | Iterable[Node],
    workers: int | None = None,
    cache: bool = True,
    dim: str | None = None,
) -> Node:

    """Aggregate multiple nodes into a single list result.

    This function creates a new Node that, when executed, will run all input
    nodes (potentially in parallel) and return a list of their results in the
    same order.

    If ``dim`` is specified, performs reduction along that dimension.

    Parameters
    ----------
    *nodes : Node | Iterable[Node]
        Nodes to gather. Can be passed as positional arguments or as a single
        iterable (list, tuple, generator).
    workers : int, optional
        Max concurrency for executing these nodes.
    cache : bool, optional
        Whether to cache the gathered result itself. Defaults to True.
    dim : str, optional
        The dimension to gather along. If provided, input must be a single
        Vector Node containing this dimension.

    Returns
    -------
    Node
        A node that evaluates to ``List[Any]``.
    """
    from .runtime import get_runtime
    from .core import Node

    if len(nodes) == 1 and not isinstance(nodes[0], Node):
        nodes_list = tuple(cast(Iterable[Node], nodes[0]))
    else:
        nodes_list = cast(tuple[Node, ...], nodes)

    # Reduction Mode
    if dim is not None:
        if len(nodes_list) != 1:
            raise ValueError("When 'dim' is specified, exactly one Vector Node must be provided.")
        target = nodes_list[0]
        if not target.dims or dim not in target.dims:
            raise ValueError(f"Dimension '{dim}' not found in target node dimensions {target.dims}.")
        
        # 1. Identify axis
        axis = target.dims.index(dim)
        
        # 2. Check if full reduction (result is Scalar)
        if len(target.dims) == 1:
            # Simple case: 1D -> Scalar
            return gather(*target._items, workers=workers, cache=cache)
        
        # 3. Partial Reduction (Result is Vector)
        # Move the target axis to the last dimension
        items_shifted = np.moveaxis(target._items, axis, -1)
        
        # New shape and dims (excluding the reduced dimension)
        new_shape = items_shifted.shape[:-1]
        new_dims = tuple(d for d in target.dims if d != dim)
        new_coords = {d: c for d, c in target.coords.items() if d != dim}
        
        # Flatten the leading dimensions to iterate easily
        # Reshape to (-1, axis_length)
        items_flat = items_shifted.reshape(-1, items_shifted.shape[-1])
        
        new_items_flat = []
        for row in items_flat:
            # Each 'row' is a 1D array of nodes to be gathered
            # Create a gather node for this slice
            reduced_node = gather(*row, workers=workers, cache=cache)
            new_items_flat.append(reduced_node)
        
        # Reshape back to new vector shape
        new_items = np.array(new_items_flat, dtype=object).reshape(new_shape)
        
        # Create output Vector Node
        result_node = Node(
            fn=gather,
            inputs={"items": target, "dim": dim},
            cache=cache,
            dims=new_dims,
            coords=new_coords,
        )
        result_node._items = new_items
        return result_node

    # Standard Gather Mode
    if not nodes_list:
        raise ValueError("no nodes provided")

    runtime = get_runtime()

    @runtime.define(workers=workers, cache=cache)
    def _gather(*items):
        return list(items)

    return _gather(*nodes_list)
