import node
from node import DiskCache


def test_dimensioned_result_repr_after_disk_cache(tmp_path):
    """DimensionedResult repr should work after DiskCache roundtrip."""
    node.reset()
    node.configure(
        cache=DiskCache(tmp_path),
        executor="thread",
        workers=1,
        continue_on_error=False,
        validate=False,
    )

    @node.dimension()
    def dim():
        return [0, 1, 2]

    @node.define()
    def f(x: int) -> int:
        return x

    # First run: compute and populate disk cache
    res1 = f(x=dim())()
    assert hasattr(res1, "dims")
    assert hasattr(res1, "coords")

    # Second run: should hit DiskCache and unpickle DimensionedResult
    res2 = f(x=dim())()

    # Ensure we got the same logical type back
    assert type(res2) is type(res1)

    # repr must not raise, and dims/coords should be preserved
    text = repr(res2)
    assert text.startswith("DimensionedResult(")
    assert res2.dims == res1.dims
    assert res2.coords == res1.coords


def test_dimensioned_result_can_skip_aggregate_cache(tmp_path):
    """Broadcast items can be cached while the collected result is not."""
    node.reset()
    disk = DiskCache(tmp_path)
    node.configure(
        cache=disk,
        executor="thread",
        workers=1,
        continue_on_error=False,
        validate=False,
    )

    @node.dimension()
    def dim():
        return [0, 1, 2]

    @node.define(cache=True, cache_aggregate=False)
    def f(x: int) -> int:
        return x + 1

    root = f(x=dim())
    result = root()

    assert hasattr(result, "dims")
    assert root.cache is False
    assert root._items is not None
    assert not disk.get(root.fn.__name__, root._hash)[0]
    assert all(
        disk.get(item.fn.__name__, item._hash)[0]
        for item in root._items.flat
    )


def test_cache_aggregate_defaults_to_cache(tmp_path):
    """When omitted, aggregate caching should follow the cache flag."""
    node.reset()
    disk = DiskCache(tmp_path)
    node.configure(
        cache=disk,
        executor="thread",
        workers=1,
        continue_on_error=False,
        validate=False,
    )

    @node.dimension()
    def dim():
        return [0, 1]

    @node.define(cache=False)
    def no_cache(x: int) -> int:
        return x

    root = no_cache(x=dim())
    root()

    assert root.cache is False
    assert root._items is not None
    assert not disk.get(root.fn.__name__, root._hash)[0]
    assert all(
        not disk.get(item.fn.__name__, item._hash)[0]
        for item in root._items.flat
    )


def test_dimension_cache_scope_defaults_to_root_only(tmp_path):
    """Default scope only caches root DimensionedResult, not intermediate ones."""
    node.reset()
    disk = DiskCache(tmp_path)
    node.configure(
        cache=disk,
        executor="thread",
        workers=1,
        continue_on_error=False,
        validate=False,
    )

    @node.dimension()
    def dim():
        return [0, 1, 2]

    @node.define()
    def a(x: int) -> int:
        return x + 1

    @node.define()
    def b(y: int) -> int:
        return y * 2

    @node.define()
    def c(z: int) -> int:
        return z - 3

    root_a = a(x=dim())
    root_b = b(y=root_a)
    root_c = c(z=root_b)
    result = root_c()
    assert hasattr(result, "dims")

    assert not disk.get(f"{root_a.fn.__name__}/dim", root_a._hash)[0]
    assert not disk.get(f"{root_b.fn.__name__}/dim", root_b._hash)[0]
    assert disk.get(f"{root_c.fn.__name__}/dim", root_c._hash)[0]


def test_dimension_cache_scope_all_caches_intermediate_dimension_results(tmp_path):
    """Scope=all preserves caching for non-root DimensionedResult nodes."""
    node.reset()
    disk = DiskCache(tmp_path)
    node.configure(
        cache=disk,
        executor="thread",
        workers=1,
        continue_on_error=False,
        validate=False,
        cache_dimension_scope="all",
    )

    @node.dimension()
    def dim():
        return [0, 1]

    @node.define()
    def a(x: int) -> int:
        return x + 1

    @node.define()
    def b(y: int) -> int:
        return y * 2

    root_a = a(x=dim())
    root_b = b(y=root_a)
    root_b()

    assert disk.get(f"{root_a.fn.__name__}/dim", root_a._hash)[0]
    assert disk.get(f"{root_b.fn.__name__}/dim", root_b._hash)[0]


