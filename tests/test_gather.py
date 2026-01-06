import pytest
from node import gather


def test_gather_util(runtime_factory):
    rt = runtime_factory()

    @rt.define()
    def inc(x):
        return x + 1

    n1 = inc(1)
    n2 = inc(2)

    result = rt.run(gather(n1, n2))
    assert result == [2, 3]

    # allow iterable input
    result2 = rt.run(gather([n1, n2]))
    assert result2 == [2, 3]


def test_gather_flow_mismatch(runtime_factory):
    flow1 = runtime_factory()
    flow2 = runtime_factory()

    @flow1.define()
    def a(x):
        return x

    @flow2.define()
    def b(x):
        return x

    with pytest.raises(ValueError):
        gather(a(1), b(2))


def test_gather_custom_workers(runtime_factory):
    rt = runtime_factory()

    @rt.define()
    def inc(x):
        return x + 1

    node = gather(inc(1), workers=3)
    assert node.fn._node_workers == 3


def test_gather_cache_toggle(runtime_factory):
    rt = runtime_factory()

    @rt.define()
    def inc(x):
        return x + 1

    node = gather(inc(1), cache=False)
    assert node.cache is False
