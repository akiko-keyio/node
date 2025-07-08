import pytest
from node import gather


def test_gather_util(flow_factory):
    flow = flow_factory()

    @flow.node()
    def inc(x):
        return x + 1

    n1 = inc(1)
    n2 = inc(2)

    result = flow.run(gather(n1, n2))
    assert result == [2, 3]

    # allow iterable input
    result2 = flow.run(gather([n1, n2]))
    assert result2 == [2, 3]


def test_gather_flow_mismatch(flow_factory):
    flow1 = flow_factory()
    flow2 = flow_factory()

    @flow1.node()
    def a(x):
        return x

    @flow2.node()
    def b(x):
        return x

    with pytest.raises(ValueError):
        gather(a(1), b(2))


def test_gather_custom_workers(flow_factory):
    flow = flow_factory()

    @flow.node()
    def inc(x):
        return x + 1

    node = gather(inc(1), workers=3)
    assert node.fn._node_workers == 3
