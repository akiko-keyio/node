"""Tests for runtime reset behavior with pre-defined nodes."""

import node
from node import Config


def test_define_uses_current_runtime_after_reset():
    """Ensure nodes defined before reset pick up new runtime defaults."""
    node.reset()
    node.configure(
        config=Config(
            {
                "global": 1,
                "my_node": {"value": "${global}"},
            }
        )
    )

    @node.define()
    def my_node(value: int) -> int:
        return value

    node.reset()
    node.configure(
        config=Config(
            {
                "global": 5,
                "my_node": {"value": "${global}"},
            }
        )
    )

    assert my_node().get() == 5
