import node
from node import DiskJoblib


def test_dimensioned_result_repr_after_disk_cache(tmp_path):
    """DimensionedResult repr should work after DiskJoblib cache roundtrip."""
    node.reset()
    node.configure(
        cache=DiskJoblib(tmp_path),
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

    # Second run: should hit DiskJoblib cache and unpickle DimensionedResult
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
    disk = DiskJoblib(tmp_path)
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
    assert not disk.contains(root.fn.__name__, root._hash)
    assert all(
        disk.contains(item.fn.__name__, item._hash)
        for item in root._items.flat
    )


def test_cache_aggregate_defaults_to_cache(tmp_path):
    """When omitted, aggregate caching should follow the cache flag."""
    node.reset()
    disk = DiskJoblib(tmp_path)
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
    assert not disk.contains(root.fn.__name__, root._hash)
    assert all(
        not disk.contains(item.fn.__name__, item._hash)
        for item in root._items.flat
    )


