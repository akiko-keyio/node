import node
from node import DiskCache
from node.dimension import DimensionedResult


def test_dimensioned_result_repr_after_disk_cache(tmp_path):
    """DimensionedResult repr should work after reading from disk cache."""
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

    @node.define(cache=True)
    def f(x: int) -> int:
        return x

    # First run: compute and populate disk cache
    res1 = f(x=dim())()
    assert hasattr(res1, "dims")
    assert hasattr(res1, "coords")

    # Second run: should hit DiskCache and unpickle DimensionedResult aggregate
    res2 = f(x=dim())()

    # Ensure we got the same logical type back
    assert type(res2) is type(res1)

    # repr must not raise, and dims/coords should be preserved
    text = repr(res2)
    assert text.startswith("DimensionedResult(")
    assert res2.dims == res1.dims
    assert res2.coords == res1.coords


def test_dimensioned_map_uses_regular_node_cache_semantics(tmp_path):
    """Dimensioned map nodes use the same cache behavior as regular nodes."""
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

    @node.define(cache=True)
    def f(x: int) -> int:
        return x + 1

    root = f(x=dim())
    result = root()

    assert hasattr(result, "dims")
    assert root.cache is True
    assert root._items is not None
    assert disk.get(root.fn.__name__, root._hash)[0]
    assert all(
        disk.get(item.fn.__name__, item._hash)[0]
        for item in root._items.flat
    )


def test_reduce_dims_data_order_follows_declaration(tmp_path):
    """reduce_dims input data should be DimensionedResult with declared dim order."""
    node.reset()
    disk = DiskCache(tmp_path)
    node.configure(
        cache=disk,
        executor="thread",
        workers=1,
        continue_on_error=False,
        validate=False,
    )

    @node.dimension(name="time")
    def time():
        return [1, 2]

    @node.dimension(name="model")
    def model():
        return ["a", "bb"]

    @node.define()
    def base(t: int, m: str) -> str:
        return f"{t}-{m}"

    @node.define(reduce_dims=["time", "model"])
    def agg(data):
        edge = f"{data[0, 0]}|{data[-1, -1]}"
        return type(data), data.dims, data.coords, data.shape, edge

    observed_type, observed_dims, observed_coords, observed_shape, edge = agg(
        data=base(t=time(), m=model())
    )()

    assert edge == "1-a|2-bb"
    assert observed_type is DimensionedResult
    assert observed_dims == ("time", "model")
    assert observed_coords["time"] == [1, 2]
    assert observed_coords["model"] == ["a", "bb"]
    assert observed_shape == (2, 2)


def test_reduce_dims_preserves_non_reduced_coord_injection(tmp_path):
    """Non-reduced dimensions should still inject the current scalar coord."""
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
    def time():
        return [0, 1, 2]

    @node.dimension()
    def model():
        return ["a", "b"]

    @node.define()
    def load(t: int, m: str) -> int:
        return t + len(m)

    @node.define(reduce_dims=["time"])
    def summary(data, model):
        # model should be scalar coord for the current non-reduced slice.
        return model, int(sum(data.flat))

    grid = load(t=time(), m=model())
    reduced = summary(data=grid)
    result = reduced()
    assert result.dims == ("model",)
    assert result.shape == (2,)
    assert result[0][0] == "a"
    assert result[1][0] == "b"
    assert result[0][1] == 6
    assert result[1][1] == 6


def test_reduce_dims_all_receives_full_dimensioned_result(tmp_path):
    node.reset()
    node.configure(
        cache=DiskCache(tmp_path),
        executor="thread",
        workers=1,
        continue_on_error=False,
        validate=False,
    )

    @node.dimension(name="time")
    def time():
        return [1, 2]

    @node.define()
    def load(t: int) -> int:
        return t * 10

    @node.define(reduce_dims="all")
    def summary(data):
        return int(sum(data.flat)), data.dims, data.coords, data.shape

    out, dims, coords, shape = summary(data=load(t=time()))()
    assert out == 30
    assert dims == ("time",)
    assert coords["time"] == [1, 2]
    assert shape == (2,)


def test_invalidate_dimension_node_clears_item_cache(tmp_path):
    """Invalidating a dimensioned node should clear reachable item caches."""
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

    @node.define()
    def a(x: int) -> int:
        return x + 1

    @node.define()
    def b(y: int) -> int:
        return y * 2

    root_a = a(x=dim())
    root_b = b(y=root_a)
    root_b()

    assert all(disk.get(item.fn.__name__, item._hash)[0] for item in root_a._items.flat)
    assert all(disk.get(item.fn.__name__, item._hash)[0] for item in root_b._items.flat)

    root_b.invalidate(recursive=True)
    assert not any(
        disk.get(item.fn.__name__, item._hash)[0] for item in root_a._items.flat
    )
    assert not any(
        disk.get(item.fn.__name__, item._hash)[0] for item in root_b._items.flat
    )


