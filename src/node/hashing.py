"""Node identity, hashing, and canonicalization utilities."""

from __future__ import annotations

import hashlib
import warnings
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .core import Node

_FAST_REPR_TYPES = frozenset({int, float, str, bool, type(None), bytes})

# id(obj) → (canonical_str, obj_ref).  Bounded to prevent unbounded growth.
_canonical_id_cache: dict[int, tuple[str, Any]] = {}
_CANONICAL_CACHE_MAX = 4096


def _is_safe_type(obj: Any, depth: int = 0) -> tuple[bool, str]:
    """Check if an object has a stable canonical representation."""
    if depth > 10:
        return False, "nested too deeply"

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
    elif isinstance(obj, np.ndarray):
        if obj.size > 1000:
            return True, ""
        if obj.dtype == object:
            for item in obj.ravel():
                safe, reason = _is_safe_type(item, depth + 1)
                if not safe:
                    return False, f"ndarray element: {reason}"

    return True, ""


def _canonical(obj: Any) -> str:
    """Convert *obj* into a deterministic string for cache-key computation.

    Dispatch order (first match wins):
    1. Primitives (int, float, str, bool, None, bytes) → ``repr()``
    2. Node → ``<fn_name>_<hex hash>``
    3. Callable with ``_node_sig`` → ``NodeFactory:<name>``
    4. Callable with ``__qualname__`` → ``Func:<qualname>``
    5. dict / list / tuple / set → recursive canonical
    6. ndarray → shape + content, with id-cache
    7. Anything else → ``repr()`` (with safety warning for exotic types)
    """
    if type(obj) in _FAST_REPR_TYPES:
        return repr(obj)

    from .core import Node

    if isinstance(obj, Node):
        return f"{obj.fn.__name__}_{obj._hash:x}"

    qname = getattr(obj, "__qualname__", None)
    if qname is not None:
        if hasattr(obj, "_node_sig"):
            return f"NodeFactory:{obj.__name__}"
        return f"Func:{qname}"

    if isinstance(obj, dict):
        inner = ", ".join(
            f"{repr(k)}: {_canonical(v)}" for k, v in sorted(obj.items())
        )
        return "{" + inner + "}"
    if isinstance(obj, (list, tuple)):
        inner = ", ".join(_canonical(v) for v in obj)
        return "[" + inner + "]" if isinstance(obj, list) else "(" + inner + ")"
    if isinstance(obj, set):
        inner = ", ".join(_canonical(v) for v in sorted(obj))
        return "{" + inner + "}"

    if isinstance(obj, np.ndarray):
        oid = id(obj)
        entry = _canonical_id_cache.get(oid)
        if entry is not None and entry[1] is obj:
            return entry[0]

        if obj.dtype == object:
            inner = ", ".join(_canonical(v) for v in obj.ravel())
            result = f"np.array(shape={obj.shape}, items=[{inner}])"
        else:
            result = f"np.array(shape={obj.shape}, data={obj.tobytes().hex()})"

        if len(_canonical_id_cache) >= _CANONICAL_CACHE_MAX:
            _canonical_id_cache.clear()
        _canonical_id_cache[oid] = (result, obj)
        return result

    safe, reason = _is_safe_type(obj)
    if not safe:
        warnings.warn(
            f"Parameter of type {type(obj).__name__} may not have stable "
            f"canonical representation: {reason}. "
            f"This could lead to unexpected cache behavior.",
            category=UserWarning,
            stacklevel=4,
        )

    return repr(obj)


def compute_node_hash(
    fn_name: str,
    inputs: dict[str, Any],
    ignore: frozenset[str] = frozenset(),
) -> int:
    """Compute a 64-bit identity hash for a node.

    The hash is derived from *fn_name* and each ``(param_name, canonical_value)``
    pair (sorted by param name), using BLAKE2b.  Node-typed values are
    represented by their own hash; all others go through ``_canonical()``.

    The resulting integer is deterministic across runs for the same inputs.
    """
    from .core import Node

    canonical_inputs = tuple(
        sorted(
            (k, f"{hash(v):016x}" if isinstance(v, Node) else _canonical(v))
            for k, v in inputs.items()
            if k not in ignore
        )
    )
    return int(
        hashlib.blake2b(
            repr((fn_name, canonical_inputs)).encode(), digest_size=8
        ).hexdigest(),
        16,
    )
