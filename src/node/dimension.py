"""Dimension-aware logic and broadcasting utilities."""

from __future__ import annotations

import itertools
from functools import wraps
from typing import Any, Callable, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .core import Node

__all__ = ["dimension", "broadcast", "DimensionedResult"]


def _get_layout_from_sig(
    fn: Callable, known_dims: set[str]
) -> tuple[list[str], list[str]]:
    """Classify parameters of *fn* into dimension args and other args.

    Uses the cached ``_node_sig`` when available to avoid repeated
    ``inspect.signature`` calls.
    """
    sig = getattr(fn, "_node_sig", None)
    if sig is None:
        import inspect

        try:
            sig = inspect.signature(fn)
        except (ValueError, TypeError):
            return [], []

    dim_args: list[str] = []
    other_args: list[str] = []
    for name, param in sig.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if name in known_dims:
            dim_args.append(name)
        else:
            other_args.append(name)
    return dim_args, other_args


def _wrap_reduced_slice(
    value: Any,
    v_dims: tuple[str, ...],
    bcast_dim_to_idx: dict[str, int],
    reduce_dims_order: tuple[str, ...],
    available_dims: dict[str, list[Any]],
) -> Any:
    """Wrap a vector slice as a DimensionedResult if reduction applies."""
    if not isinstance(value, np.ndarray) or value.ndim == 0:
        return value
    sliced_dims = tuple(d for d in v_dims if d not in bcast_dim_to_idx)
    reduced_dims = tuple(d for d in reduce_dims_order if d in sliced_dims)
    if not reduced_dims:
        return value
    if sliced_dims != reduced_dims:
        perm = tuple(sliced_dims.index(d) for d in reduced_dims)
        value = np.transpose(value, perm)
    return DimensionedResult(
        value,
        dims=reduced_dims,
        coords={d: available_dims[d] for d in reduced_dims},
    )


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
    reduce_dims_order: tuple[str, ...] | None = None,
) -> tuple[tuple[str, ...], dict[str, list[Any]], np.ndarray]:
    """Broadcast vector inputs into scalar Node instances.

    When *target_dims* is ``None`` (pure map), all available dimensions are
    broadcast.  When *target_dims* is a subset, the remaining dimensions are
    left as full slices and optionally wrapped with *reduce_dims_order*.
    """
    from .core import Node

    # 1. Collect dimensions & validate
    available_dims: dict[str, list[Any]] = {}
    seen: set[str] = set()
    for node in vector_inputs.values():
        for d in node.dims:
            if d not in seen:
                seen.add(d)
                available_dims[d] = node.coords[d]
            else:
                existing = available_dims[d]
                if existing is not node.coords[d]:
                    eq = (
                        np.array_equal(node.coords[d], existing)
                        if isinstance(node.coords[d], np.ndarray)
                        or isinstance(existing, np.ndarray)
                        else node.coords[d] == existing
                    )
                    if not eq:
                        from .exceptions import DimensionMismatchError

                        raise DimensionMismatchError(
                            f"Dimension Mismatch: '{d}' conflict."
                        )

    # 2. Determine broadcast dimensions
    broadcast_dims = (
        tuple(sorted(available_dims)) if target_dims is None else target_dims
    )
    shape = tuple(len(available_dims[d]) for d in broadcast_dims)

    dims_to_inject, _ = _get_layout_from_sig(fn, set(available_dims))

    # 3. Pre-compute index mapping
    bcast_dim_to_idx: dict[str, int] = {
        d: i for i, d in enumerate(broadcast_dims)
    }
    vector_dim_positions = {
        k: [bcast_dim_to_idx.get(d) for d in v.dims]
        for k, v in vector_inputs.items()
    }
    base_inputs = {
        k: v
        for k, v in inputs.items()
        if not (isinstance(v, Node) and v.dims)
    }

    # 4. Separate constant vs varying vector inputs
    template = dict(base_inputs)
    varying_vector_keys: list[str] = []

    for k, v in vector_inputs.items():
        positions = vector_dim_positions[k]
        if all(pos is None for pos in positions):
            value = v._items[tuple(slice(None) for _ in positions)]
            if reduce_dims_order is not None:
                value = _wrap_reduced_slice(
                    value, v.dims, bcast_dim_to_idx,
                    reduce_dims_order, available_dims,
                )
            template[k] = value
        else:
            varying_vector_keys.append(k)

    # Constant coordinate injections (non-broadcast dims)
    varying_coord_dims: list[str] = []
    for d in dims_to_inject:
        if d in bcast_dim_to_idx:
            varying_coord_dims.append(d)
        elif d not in template:
            template[d] = available_dims[d]

    # 5. Pre-compute deps from the template (scanned once, reused every iteration)
    from .core import _collect_nodes

    template_deps = _collect_nodes(template.values())

    # 6. Expand
    items_flat: list[Node] = []
    for indices in itertools.product(*(range(s) for s in shape)):
        scalar_inputs = dict(template)
        iter_deps = list(template_deps)

        for k in varying_vector_keys:
            v = vector_inputs[k]
            positions = vector_dim_positions[k]
            slice_key = tuple(
                indices[pos] if pos is not None else slice(None)
                for pos in positions
            )
            value = v._items[slice_key]
            if reduce_dims_order is not None:
                value = _wrap_reduced_slice(
                    value, v.dims, bcast_dim_to_idx,
                    reduce_dims_order, available_dims,
                )
            scalar_inputs[k] = value
            if isinstance(value, Node):
                iter_deps.append(value)
            elif isinstance(value, np.ndarray) and value.dtype == np.object_:
                for x in value.flat:
                    if isinstance(x, Node):
                        iter_deps.append(x)

        for d in varying_coord_dims:
            if d not in scalar_inputs:
                scalar_inputs[d] = available_dims[d][indices[bcast_dim_to_idx[d]]]

        items_flat.append(
            Node(fn=fn, inputs=scalar_inputs, cache=cache, _deps=iter_deps)
        )

    items_array = np.array(items_flat, dtype=object).reshape(shape)
    out_coords = {d: available_dims[d] for d in broadcast_dims}
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
        obj = np.asarray(input_array).view(cls)
        obj.dims = dims
        obj.coords = coords or {}
        return obj

    def __array_finalize__(self, obj: np.ndarray | None) -> None:
        if obj is None:
            return
        self.dims = getattr(obj, "dims", ())
        self.coords = getattr(obj, "coords", {})

    def __reduce__(self):
        func, args, state = super().__reduce__()
        new_state = (state, getattr(self, "dims", ()), getattr(self, "coords", {}))
        return func, args, new_state

    def __setstate__(self, state):
        if isinstance(state, tuple) and len(state) == 3:
            ndarray_state, dims, coords = state
            super().__setstate__(ndarray_state)
            self.dims = dims
            self.coords = coords
            return
        super().__setstate__(state)
        if not hasattr(self, "dims"):
            self.dims = ()
        if not hasattr(self, "coords"):
            self.coords = {}

    def transpose(self, *order: str) -> "DimensionedResult":
        """Transpose axes by dimension names."""
        if set(order) != set(self.dims):
            raise ValueError(f"Order {order} doesn't match dims {self.dims}")
        perm = tuple(self.dims.index(d) for d in order)
        result = super().transpose(perm).view(DimensionedResult)
        result.dims = order
        result.coords = self.coords
        return result

    def __repr__(self) -> str:
        dims = getattr(self, "dims", ())
        coords = getattr(self, "coords", {})
        dims_str = ", ".join(f"{d}:{len(coords.get(d, []))}" for d in dims)
        return f"DimensionedResult({super().__repr__()}, dims=({dims_str}))"
