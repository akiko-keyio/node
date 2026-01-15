"""Node identity, hashing, and canonicalization utilities."""
from __future__ import annotations

import hashlib
import warnings
from typing import Any, TYPE_CHECKING
from collections.abc import Callable

if TYPE_CHECKING:
    from .core import Node

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
    
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            if obj.size > 1000: # Sanity limit for DAG objects
                return True, "" # Assume safe if too big (pure data)
            if obj.dtype == object:
                for item in obj.ravel():
                    safe, reason = _is_safe_type(item, depth + 1)
                    if not safe:
                        return False, f"ndarray element: {reason}"
    except ImportError:
        pass
    
    return True, ""


def _canonical(obj: Any) -> str:
    """Convert an object into a deterministic string representation."""
    from .core import Node
    if isinstance(obj, Node):
        return f"{obj.fn.__name__}_{obj._hash:x}"
    
    # Custom node definitions
    try:
        obj._node_sig
        return f"NodeFactory:{obj.__name__}"
    except AttributeError:
        pass
    
    # Regular functions/methods: avoid memory addresses in hash
    try:
        return f"Func:{obj.__qualname__}"
    except AttributeError:
        pass

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
    
    # Numpy support
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
             # Hash the shape and content
             # converting to list is safe but slow for huge arrays.
             # For DAG construction (meta-data), arrays are usually small (coords or node lists).
             # If array contains Nodes, we must recurse.
             if obj.dtype == object:
                 flat = obj.ravel()
                 inner = ", ".join(_canonical(v) for v in flat)
                 return f"np.array(shape={obj.shape}, items=[{inner}])"
             else:
                 # Primitive array
                 return f"np.array(shape={obj.shape}, data={obj.tobytes().hex()})"
    except ImportError:
        pass

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
    from .core import Node
    
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
