"""Dimension-aware logic and broadcasting utilities."""
from __future__ import annotations

import itertools
from functools import wraps
from typing import Any, Callable, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .core import Node

def dimension(name: str | None = None):
    """Decorator to define a dimension node.
    
    The decorated function must return a list of values.
    
    Args:
        name: Optional name for the dimension. Defaults to function name.
        
    Returns:
        A decorator that converts the function into a Vector Node factory.
    """
    import inspect
    from .core import Node
    
    def decorator(fn: Callable):
        sig = inspect.signature(fn)
        
        @wraps(fn)
        def wrapper(*args, **kwargs):
            # 1. Eager Execution
            raw_values = fn(*args, **kwargs)
            if not isinstance(raw_values, list):
                raw_values = list(raw_values)
            
            # 2. Vector Encapsulation
            dim_name = name or fn.__name__
            
            # Bind arguments for logical inputs
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            logical_inputs = bound.arguments
            
            # Create Vector Node
            # The Vector Node represents the generator function call
            vector_node = Node(
                fn=fn,
                inputs=logical_inputs,
                cache=True, 
                dims=(dim_name,),
                coords={dim_name: raw_values},
            )
            # Store raw values directly (optimization: no _identity wrapper)
            vector_node._items = np.array(raw_values, dtype=object)
            
            return vector_node
            
        return wrapper
    return decorator


def broadcast(
    fn: Callable,
    inputs: dict[str, Any],
    vector_inputs: dict[str, "Node"],
    cache: bool,
) -> tuple[tuple[str, ...], dict[str, list[Any]], np.ndarray]:
    """Perform broadcasting logic for Vector Nodes.
    
    Args:
        fn: The function being called.
        inputs: All inputs to the function (mixed scalar/vector).
        vector_inputs: Subset of inputs that are Vector Nodes.
        cache: Cache setting for generated scalar nodes.
        
    Returns:
        Tuple of (sorted_dims, coord_dict, items_array).
    """
    from .core import Node
    
    # 1. Collect Dimensions & Validate (Identity Check)
    all_dims: dict[str, list[Any]] = {}  # dim_name -> coords
    
    seen_dims = set()
    for node in vector_inputs.values():
        for d in node.dims:
            if d not in seen_dims:
                seen_dims.add(d)
                all_dims[d] = node.coords[d] 
            else:
                if all_dims[d] is not node.coords[d]:
                    raise ValueError(
                        f"Dimension mismatch: '{d}' found with different coordinate instances."
                    )
    
    sorted_dims = tuple(sorted(all_dims.keys()))
    shape = tuple(len(all_dims[d]) for d in sorted_dims)
    
    # 2. Virtual Expansion
    ranges = [range(s) for s in shape]
    items_flat = []
    dim_to_idx = {d: i for i, d in enumerate(sorted_dims)}
    
    for indices in itertools.product(*ranges):
        scalar_inputs = {}
        for arg_name, arg_val in inputs.items():
            if isinstance(arg_val, Node) and arg_val.dims:
                node_indices = []
                for d in arg_val.dims:
                    global_idx_pos = dim_to_idx[d]
                    idx_val = indices[global_idx_pos]
                    node_indices.append(idx_val)
                item = arg_val._items[tuple(node_indices)]
                scalar_inputs[arg_name] = item
            else:
                scalar_inputs[arg_name] = arg_val
        
        scalar_node = Node(fn=fn, inputs=scalar_inputs, cache=cache)
        items_flat.append(scalar_node)
        
    items_array = np.array(items_flat, dtype=object).reshape(shape)
    return sorted_dims, all_dims, items_array
