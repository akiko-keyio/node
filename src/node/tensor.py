"""Logical tensor orchestration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from .core import Node, gather

__all__ = ["Dim", "TensorNode"]


@dataclass(frozen=True)
class Dim:
    """Dimension marker for tensor orchestration.

    Attributes:
        name: Dimension name used for alignment.
        values: Ordered coordinate values for this dimension.
    """

    name: str
    values: tuple[Any, ...]

    def __init__(self, name: str, values: Iterable[Any]):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "values", tuple(values))

    def __len__(self) -> int:
        return len(self.values)


class TensorNode:
    """Logical container for a 1D tensor of Nodes.

    This is the first step toward tensor orchestration. It wraps a sequence of
    Nodes and executes them via the existing runtime using ``gather``.
    """

    __slots__ = ("_nodes", "_dim")

    def __init__(self, nodes: Sequence[Node], *, dim: Dim | None = None):
        self._nodes = tuple(nodes)
        self._dim = dim

    @property
    def nodes(self) -> tuple[Node, ...]:
        """Return the underlying Node sequence."""
        return self._nodes

    @property
    def dim(self) -> Dim | None:
        """Return the dimension metadata for this tensor."""
        return self._dim

    @classmethod
    def from_nodes(cls, nodes: Sequence[Node], *, dim: Dim | None = None) -> "TensorNode":
        """Create a TensorNode from a node sequence.

        Args:
            nodes: Sequence of Node objects.
            dim: Optional dimension metadata. Length must match nodes.

        Returns:
            A TensorNode wrapping the provided nodes.
        """
        if not nodes:
            raise ValueError("nodes must not be empty")
        if dim is not None and len(dim) != len(nodes):
            raise ValueError("dim length must match nodes length")
        return cls(nodes, dim=dim)

    def apply(self, fn: Callable[..., Node], *others: Any) -> "TensorNode":
        """Apply a node factory across this tensor with optional broadcasting.

        Args:
            fn: A node-decorated function returning Node.
            others: Other TensorNode instances or scalars to broadcast.

        Returns:
            A new TensorNode holding the applied Nodes.
        """
        length = len(self._nodes)
        for other in others:
            if isinstance(other, TensorNode):
                if len(other.nodes) != length:
                    raise ValueError("dim length must match nodes length")
                if self._dim is not None and other.dim is not None:
                    if self._dim.name != other.dim.name:
                        raise ValueError("dim name must match")
                    if self._dim.values != other.dim.values:
                        raise ValueError("dim values must match")
        result_nodes: list[Node] = []
        for idx in range(length):
            args = [self._nodes[idx]]
            for other in others:
                if isinstance(other, TensorNode):
                    args.append(other.nodes[idx])
                else:
                    args.append(other)
            node = fn(*args)
            if not isinstance(node, Node):
                raise TypeError("apply expects a node factory that returns Node")
            result_nodes.append(node)
        return TensorNode.from_nodes(result_nodes, dim=self._dim)

    def __call__(self) -> list[Any]:
        """Execute all nodes and return results in order."""
        return gather(self._nodes)()
