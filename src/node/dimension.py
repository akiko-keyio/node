"""Dimension-aware logic and broadcasting utilities."""
from __future__ import annotations

import itertools
import inspect
from functools import wraps
from typing import Any, Callable, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .core import Node

__all__ = ["dimension", "broadcast", "DimensionedResult"]


def _get_layout_from_sig(fn: Callable, known_dims: set[str]) -> tuple[list[str], list[str]]:
    """Determine expected dimension layout from function signature."""
    try:
        sig = inspect.signature(fn)
    except ValueError:
         return [], []

    dim_args = []
    other_args = []
    
    for name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
            
        if name in known_dims:
            dim_args.append(name)
        else:
            other_args.append(name)
            
    return dim_args, other_args




def dimension(name: str | None = None):
    """Decorator to define a dimension node."""
    import inspect
    from .core import Node
    
    def decorator(fn: Callable):
        sig = inspect.signature(fn)
        
        @wraps(fn)
        def wrapper(*args, **kwargs):
            raw_values = fn(*args, **kwargs)
            if not isinstance(raw_values, list):
                raw_values = list(raw_values)
            
            dim_name = name or fn.__name__
            
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            node = Node(
                fn=fn,
                inputs=bound.arguments,
                cache=True, 
                dims=(dim_name,),
                coords={dim_name: raw_values},
            )
            node._items = np.array(raw_values, dtype=object)
            return node
            
        return wrapper
    return decorator


def broadcast(
    fn: Callable,
    inputs: dict[str, Any],
    vector_inputs: dict[str, "Node"],
    cache: bool,
    target_dims: tuple[str, ...] | None = None,
) -> tuple[tuple[str, ...], dict[str, list[Any]], np.ndarray]:
    """Perform broadcasting logic for Vector Nodes with Partial Slicing support.
    
    Args:
        fn: The function being called.
        inputs: All inputs.
        vector_inputs: Subset of inputs that are Vector Nodes.
        cache: Cache setting.
        target_dims: If provided, specific dimensions to broadcast over. 
                     Dimensions in inputs but NOT in target_dims are passed as full slices (vectors).
                     If None, defaults to Union of all input dims.
    """
    from .core import Node
    
    # 1. Collect Dimensions & Validate
    available_dims: dict[str, list[Any]] = {} 
    
    seen = set()
    for node in vector_inputs.values():
        for d in node.dims:
            if d not in seen:
                seen.add(d)
                available_dims[d] = node.coords[d] 
            else:
                existing = available_dims[d]
                if existing is not node.coords[d]:
                     is_equal = np.array_equal(node.coords[d], existing) if (isinstance(node.coords[d], np.ndarray) or isinstance(existing, np.ndarray)) else (node.coords[d] == existing)
                     
                     if not is_equal:
                        from .exceptions import DimensionMismatchError
                        raise DimensionMismatchError(f"Dimension Mismatch: '{d}' conflict.")
    
    # 2. Determine Broadcast Dimensions
    if target_dims is None:
        broadcast_dims = tuple(sorted(available_dims.keys()))
    else:
        broadcast_dims = target_dims
        
    shape = tuple(len(available_dims[d]) for d in broadcast_dims)
    
    # Parse signature to see which coords need injection
    # We look for ANY dimension available
    dims_to_inject, _ = _get_layout_from_sig(fn, set(available_dims.keys()))
    
    # 3. Virtual Expansion
    ranges = [range(s) for s in shape]
    items_flat = []
    
    # Pre-compute indices mapping
    bcast_dim_to_idx = {d: i for i, d in enumerate(broadcast_dims)}
    
    for indices in itertools.product(*ranges):
        scalar_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, Node) and v.dims:
                # determine slice
                slice_key = []
                for d in v.dims:
                    if d in bcast_dim_to_idx:
                        idx_pos = bcast_dim_to_idx[d]
                        slice_key.append(indices[idx_pos])
                    else:
                        slice_key.append(slice(None))
                
                try:
                    item = v._items[tuple(slice_key)]
                    scalar_inputs[k] = item
                except Exception as e:
                    # Fallback or re-raise
                    raise e
            else:
                scalar_inputs[k] = v
        
        # Inject Current Coordinates (Implicit Args)
        for d in dims_to_inject:
             if d in scalar_inputs:
                 continue
                 
             if d in bcast_dim_to_idx:
                 idx = indices[bcast_dim_to_idx[d]]
                 scalar_inputs[d] = available_dims[d][idx]
             else:
                 # It is not broadcast, so pass full vector (gather)
                 scalar_inputs[d] = available_dims[d]

        items_flat.append(Node(fn=fn, inputs=scalar_inputs, cache=cache))
        
    items_array = np.array(items_flat, dtype=object).reshape(shape)
    
    # Filter coords
    out_coords = {d: available_dims[d] for d in broadcast_dims if d in available_dims}
    
    return broadcast_dims, out_coords, items_array


class DimensionedResult(np.ndarray):
    """A numpy array subclass that carries dimension metadata.
    
    Attributes:
        dims: Tuple of dimension names, ordered to match array axes.
        coords: Dict mapping dimension names to coordinate values.
    """
    
    dims: tuple[str, ...]
    coords: dict[str, list[Any]]
    
    def __new__(
        cls,
        input_array: np.ndarray,
        dims: tuple[str, ...] = (),
        coords: dict[str, list[Any]] | None = None,
    ) -> "DimensionedResult":
        """Create a new DimensionedResult instance.
        
        Args:
            input_array: The underlying numpy array.
            dims: Dimension names in axis order.
            coords: Coordinate values for each dimension.
        """
        obj = np.asarray(input_array).view(cls)
        obj.dims = dims
        obj.coords = coords or {}
        return obj
    
    def __array_finalize__(self, obj: np.ndarray | None) -> None:
        """Preserve attributes when array operations create new instances."""
        if obj is None:
            return
        self.dims = getattr(obj, "dims", ())
        self.coords = getattr(obj, "coords", {})
    
    def transpose(self, *order: str) -> "DimensionedResult":
        """Transpose axes by dimension names.
        
        Args:
            *order: Dimension names in desired order.
        
        Returns:
            Transposed DimensionedResult with updated dims.
        
        Raises:
            ValueError: If order doesn't match current dims.
        """
        if set(order) != set(self.dims):
            raise ValueError(
                f"Order {order} doesn't match dims {self.dims}"
            )
        
        # Compute axis permutation
        perm = tuple(self.dims.index(d) for d in order)
        
        # Transpose data
        result = super().transpose(perm).view(DimensionedResult)
        result.dims = order
        result.coords = self.coords  # Coords dict unchanged, keys still valid
        return result
    
    def __repr__(self) -> str:
        dims_str = ", ".join(f"{d}:{len(self.coords.get(d, []))}" for d in self.dims)
        return f"DimensionedResult({super().__repr__()}, dims=({dims_str}))"
