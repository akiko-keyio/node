"""Regression tests for VectorNode execution in parallel scheduling and reporter callbacks.

Covers:
  1. VectorNodes bypassed in process pool (now routed through _eval_node locally)
  2. Process pool cache hits didn't submit downstream nodes
  3. Process pool cache hits didn't store results in _results
  4. VectorNode assembly incorrectly reported cached=True
  5. VectorNode assembly result now cached (build_graph pruning)
"""
import pytest
import numpy as np
import node
from node import MemoryLRU


# ---------------------------------------------------------------------------
# Bug 1: VectorNodes must be assembled locally, never sent to a pool
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_vectornode_broadcast_parallel(runtime_factory):
    """VectorNode broadcasting works with workers > 1 (thread pool)."""
    rt = runtime_factory(workers=2, cache=MemoryLRU())

    @node.dimension(name="idx")
    def indices():
        return [1, 2, 3]

    @node.define()
    def double(x):
        return x * 2

    idx = indices()
    doubled = double(x=idx)
    result = rt.run(doubled)

    assert result.dims == ("idx",)
    assert list(result.flat) == [2, 4, 6]


@pytest.mark.unit
def test_vectornode_reduction_parallel(runtime_factory):
    """Reduction VectorNodes work correctly with parallel execution."""
    rt = runtime_factory(workers=2, cache=MemoryLRU())

    @node.dimension(name="idx")
    def indices():
        return [1, 2, 3]

    @node.define()
    def double(x):
        return x * 2

    @node.define(reduce_dims="all")
    def total(data):
        return sum(data.flat)

    idx = indices()
    doubled = double(x=idx)
    root = total(doubled)
    result = rt.run(root)

    assert result == 12  # (1*2 + 2*2 + 3*2)


@pytest.mark.unit
def test_vectornode_2d_broadcast_parallel(runtime_factory):
    """2D broadcasting works with parallel execution."""
    rt = runtime_factory(workers=2, cache=MemoryLRU())

    @node.dimension(name="time")
    def time_gen():
        return [1, 2]

    @node.dimension(name="model")
    def model_gen():
        return ["A", "B"]

    @node.define()
    def predict(t, m):
        return f"{m}_{t}"

    ts = time_gen()
    ms = model_gen()
    grid = predict(t=ts, m=ms)
    result = rt.run(grid)

    assert result.dims == ("model", "time")
    assert result.shape == (2, 2)
    assert result[0, 0] == "A_1"
    assert result[1, 1] == "B_2"


@pytest.mark.unit
def test_vectornode_process_pool(runtime_factory):
    """VectorNodes work correctly when executor='process'."""
    rt = runtime_factory(executor="process", workers=2, cache=MemoryLRU())

    @node.dimension(name="idx")
    def indices():
        return [10, 20]

    @node.define()
    def inc(x):
        return x + 1

    @node.define(reduce_dims="all")
    def total(data):
        return sum(data.flat)

    idx = indices()
    result = rt.run(total(inc(x=idx)))

    assert result == 32  # (10+1) + (20+1) = 32


# ---------------------------------------------------------------------------
# Bug 2 + 3: Process pool cache hit must submit downstream and store results
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_cache_hit_submits_downstream_parallel(runtime_factory):
    """Cache hits in parallel scheduling must submit ready downstream nodes."""
    cache = MemoryLRU()
    rt = runtime_factory(workers=2, cache=cache)

    @node.define()
    def base(x):
        return x + 1

    @node.define()
    def downstream(y):
        return y * 2

    b = base(x=5)
    root = downstream(y=b)

    # First run: both nodes execute
    result1 = rt.run(root)
    assert result1 == 12  # (5+1)*2

    # Invalidate only downstream; base stays cached
    cache.delete(root.fn.__name__, root._hash)

    # Second run: base hits cache, downstream must still execute
    result2 = rt.run(root)
    assert result2 == 12


@pytest.mark.unit
def test_cache_hit_stores_in_results(runtime_factory):
    """Cache hits must store result in _results for downstream resolution."""
    cache = MemoryLRU()
    rt = runtime_factory(workers=2, cache=cache)

    @node.define()
    def compute(x):
        return x + 1

    @node.define()
    def passthrough(y):
        return y

    n = compute(x=5)
    chain = passthrough(y=n)

    # First run populates cache for both
    result1 = rt.run(chain)
    assert result1 == 6

    # Invalidate only passthrough; compute stays cached
    cache.delete(chain.fn.__name__, chain._hash)

    # Second run: compute hits cache and must store in _results
    result2 = rt.run(chain)
    assert result2 == 6
    assert n._hash in rt._results


# ---------------------------------------------------------------------------
# Bug 4: VectorNode on_node_end must report cached=False
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_vectornode_on_node_end_not_cached(runtime_factory):
    """VectorNode assembly should report cached=False, not cached=True."""
    rt = runtime_factory(cache=MemoryLRU(), workers=1)
    events = []

    def on_node_end(n, dur, cached, failed):
        events.append((n.fn.__name__, cached, failed))

    rt.on_node_end = on_node_end

    @node.dimension(name="idx")
    def indices():
        return [1, 2]

    @node.define()
    def double(x):
        return x * 2

    idx = indices()
    root = double(x=idx)
    rt.run(root)

    # On first run, no node should report cached=True
    for name, cached, failed in events:
        if not failed:
            assert cached is False, (
                f"Node '{name}' incorrectly reported cached=True on first run"
            )


# ---------------------------------------------------------------------------
# Bug 5: Pickle fallback must save result (not silently discard it)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_process_pool_pickle_fallback_saves_result(runtime_factory):
    """When subprocess result can't be unpickled, local re-execution result must be saved."""
    rt = runtime_factory(executor="process", workers=2, cache=MemoryLRU())

    @node.define()
    def make_closure(x):
        # Closures may fail to unpickle from subprocess depending on the serializer.
        # Even if loky/cloudpickle handles it, the code path must be correct.
        def inner():
            return x
        return inner

    n = make_closure(x=42)
    result = rt.run(n)

    # Result must not be None — it should be the closure
    assert result is not None
    assert n._hash in rt._results


# ---------------------------------------------------------------------------
# VectorNode assembly result caching
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_vectornode_result_cached(runtime_factory):
    """VectorNode assembly result is written to cache so build_graph can prune it."""
    cache = MemoryLRU()
    rt = runtime_factory(workers=1, cache=cache)

    @node.dimension(name="idx")
    def indices():
        return [1, 2, 3]

    @node.define()
    def double(x):
        return x * 2

    idx = indices()
    doubled = double(x=idx)

    result1 = rt.run(doubled)
    assert list(result1.flat) == [2, 4, 6]

    # VectorNode result should now be in cache
    hit, val = cache.get(doubled.fn.__name__, doubled._hash)
    assert hit, "VectorNode assembly result was not cached"
    assert list(val.flat) == [2, 4, 6]


@pytest.mark.unit
def test_vectornode_cache_hit_skips_subnodes(runtime_factory):
    """Second run of a cached VectorNode should not re-execute sub-nodes."""
    cache = MemoryLRU()
    rt = runtime_factory(workers=1, cache=cache)

    call_count = 0

    @node.dimension(name="idx")
    def indices():
        return [10, 20]

    @node.define()
    def inc(x):
        nonlocal call_count
        call_count += 1
        return x + 1

    idx = indices()
    root = inc(x=idx)

    # First run: inc called twice (once per dimension value)
    rt.run(root)
    assert call_count == 2

    # Second run: VectorNode hits cache in build_graph, sub-nodes pruned
    call_count = 0
    result = rt.run(root)
    assert call_count == 0, f"Sub-nodes re-executed {call_count} times, expected 0"
    assert list(result.flat) == [11, 21]


@pytest.mark.unit
def test_dimension_node_result_cached(runtime_factory):
    """Dimension definition node (raw _items) result is also cached."""
    cache = MemoryLRU()
    rt = runtime_factory(workers=1, cache=cache)

    @node.dimension(name="idx")
    def indices():
        return [1, 2, 3]

    idx = indices()
    result = rt.run(idx)

    # Dimension node result should be in cache
    hit, val = cache.get(idx.fn.__name__, idx._hash)
    assert hit, "Dimension node result was not cached"
    assert val.dims == ("idx",)
    assert list(val.flat) == [1, 2, 3]


@pytest.mark.unit
def test_vectornode_cache_with_reduction(runtime_factory):
    """Reduction VectorNode result is cached and reused."""
    cache = MemoryLRU()
    rt = runtime_factory(workers=2, cache=cache)

    call_count = 0

    @node.dimension(name="idx")
    def indices():
        return [1, 2, 3]

    @node.define()
    def double(x):
        nonlocal call_count
        call_count += 1
        return x * 2

    @node.define(reduce_dims="all")
    def total(data):
        return sum(data.flat)

    idx = indices()
    root = total(double(x=idx))

    result1 = rt.run(root)
    assert result1 == 12

    # Second run: all intermediate VectorNodes cached, no re-execution
    call_count = 0
    result2 = rt.run(root)
    assert result2 == 12
    assert call_count == 0, f"Sub-nodes re-executed {call_count} times, expected 0"
