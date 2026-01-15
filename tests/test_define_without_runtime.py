"""Tests for define decorator without Runtime configured."""

import pytest
import node


def test_define_without_runtime():
    """Ensure @node.define() works without Runtime configured."""
    node.reset()
    
    # Decoration should succeed without Runtime
    @node.define()
    def task(x: int) -> int:
        return x * 2
    
    # Calling the factory should fail without Runtime
    with pytest.raises(RuntimeError, match="Runtime is not configured"):
        task(5)


def test_define_then_configure():
    """Ensure decorated functions work after Runtime is configured."""
    node.reset()
    
    @node.define()
    def task(x: int) -> int:
        return x * 2
    
    node.configure()
    result = task(5)()
    assert result == 10
