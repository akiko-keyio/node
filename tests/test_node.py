"""
Comprehensive test-suite for `zen_flow.py` (rev-6, 2025-05-31)
-----------------------------------------------------------------
Run with `pytest -q`.  The tests cover:
  • Basic linear DAG execution
  • Diamond (fan-out/fan-in) dependency sharing & cache promotion
  • Two-run cache reuse (cold → warm)
  • Canonicalisation of `set` arguments
  • Cycle detection guard
  • __repr__ code generation
  • Optional ProcessPool backend on POSIX systems
"""

import os
import pytest
import importlib

# Import the module under test
ZF = importlib.import_module("zen_flow")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fresh_flow(**kwargs):
    """Return a brand-new Flow with deterministic, in-memory cache only."""
    cache = ZF.ChainCache([ZF.MemoryLRU(maxsize=128)])
    return ZF.Flow(cache=cache, log=False, **kwargs)

# ---------------------------------------------------------------------------
# 1. Linear chain (example from __main__)
# ---------------------------------------------------------------------------
def test_linear_chain_execution_and_count():
    flow = fresh_flow()

    @flow.task()
    def add(x, y):
        return x + y

    @flow.task()
    def square(z):
        return z * z

    node = square(add(2, 3))
    result = flow.run(node)
    assert result == 25
    # Two real evaluations: add + square
    assert flow.engine._exec_count == 2

# ---------------------------------------------------------------------------
# 2. __repr__ code generation outputs valid instantiation script
# ---------------------------------------------------------------------------
def test_repr_generates_valid_script():
    flow = fresh_flow()

    @flow.task()
    def add(x, y):
        return x + y

    @flow.task()
    def square(z):
        return z * z

    node = square(add(2, 3))
    script = repr(node)
    # Should define n0 for add, n1 for square, and return n1
    expected_lines = [
        "n0 = add(x=2, y=3)",
        "n1 = square(z=n0)",
        "n1"
    ]
    assert script.splitlines() == expected_lines

    # Executing the script in a context with functions defined yields same Node
    local_vars = {"add": add, "square": square}
    exec(script, {}, local_vars)
    # After exec, n1 should be a Node with same signature
    n1 = local_vars.get("n1")
    assert isinstance(n1, ZF.Node)
    assert n1.signature == node.signature

# ---------------------------------------------------------------------------
# 3. Diamond dependency
# ---------------------------------------------------------------------------
def test_diamond_dependency():
    flow = fresh_flow()

    @flow.task()
    def base(x):
        return x * 2

    @flow.task()
    def left(b):
        return b + 1

    @flow.task()
    def right(b):
        return b + 2

    @flow.task()
    def combine(l, r):
        return l * r

    root = combine(left(base(5)), right(base(5)))
    val = flow.run(root)

    # base(5) = 10 → left = 11, right = 12 → combine = 132
    assert val == 132

    # Nodes materialised once each: base, left, right, combine = 4
    assert flow.engine._exec_count == 4

# ---------------------------------------------------------------------------
# 4. Cache reuse between runs (warm cache)
# ---------------------------------------------------------------------------
def test_cache_reuse():
    flow = fresh_flow()

    @flow.task()
    def slow_inc(x):
        return x + 1

    root = slow_inc(41)
    first = flow.run(root)
    assert first == 42
    assert flow.engine._exec_count == 1

    # Warm run: nothing executed
    second = flow.run(root)
    assert second == 42
    assert flow.engine._exec_count == 0

# ---------------------------------------------------------------------------
# 5. Canonicalisation of `set` arguments → identical signatures
# ---------------------------------------------------------------------------
def test_set_canonicalisation():
    flow = fresh_flow()

    @flow.task()
    def echo(x):
        return x

    node_a = echo({3, 1, 2})
    node_b = echo({2, 3, 1})
    assert node_a is node_b

    out = flow.run(node_a)
    assert out == {1, 2, 3}

# ---------------------------------------------------------------------------
# 6. Cycle detection guard raises immediately
# ---------------------------------------------------------------------------
def test_cycle_detection():
    flow = fresh_flow()

    @flow.task()
    def identity(x):
        return x

    # Manually inject a self-dependency and expect ValueError
    with pytest.raises(ValueError, match="Cycle detected"):
        a = identity(0)
        a.args = (a,)
        flow.run(a)

# ---------------------------------------------------------------------------
# 7. Optional ProcessPool backend (POSIX only, skipped on Windows)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(os.name == "nt", reason="ProcessPool unsupported on Windows in zen_flow rev-6")
def test_process_backend():
    flow = fresh_flow(executor="process", workers=2)

    @flow.task()
    def mul(x, y):
        return x * y

    res = flow.run(mul(6, 7))
    assert res == 42
    assert flow.engine._exec_count == 1
