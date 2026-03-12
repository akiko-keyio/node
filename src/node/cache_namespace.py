"""Shared cache namespace helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Node


def cache_namespace(node: "Node") -> str:
    """Return cache namespace for a node."""
    return f"{node.fn.__name__}/dim" if node.dims else node.fn.__name__
