from node.node import Node, Flow, ChainCache, MemoryLRU, DiskJoblib


def test_repr_matches_signature(tmp_path):
    flow = Flow(cache=ChainCache([MemoryLRU(), DiskJoblib(tmp_path)]), log=False)

    @flow.node()
    def add(x, y):
        return x + y

    @flow.node()
    def square(z):
        return z * z

    linear = square(add(2, 3))
    assert repr(linear) == linear.signature

    diamond = add(add(1, 2), add(1, 2))
    assert repr(diamond) == diamond.signature


def test_branch_no_diamond(tmp_path):
    """Branching without shared nodes should not expand to a script."""
    flow = Flow(cache=ChainCache([MemoryLRU(), DiskJoblib(tmp_path)]), log=False)

    @flow.node()
    def add(x, y):
        return x + y

    @flow.node()
    def square(z):
        return z * z

    node = square(add(square(1), square(2)))
    expected = "square(z=add(x=square(z=1), y=square(z=2)))"
    assert node.signature == expected


def test_signature_key_canonicalization():
    def identity(x=None, **kw):
        return (x, kw)

    n1 = Node(identity, ({2, 1},))
    n2 = Node(identity, ({1, 2},))

    assert n1.signature == n2.signature

    d1 = Node(identity, kwargs={"d": {"b": 2, "a": 1}})
    d2 = Node(identity, kwargs={"d": {"a": 1, "b": 2}})

    assert d1.signature == d2.signature


def test_signature_script_dedup():
    def add(x, y):
        return x + y

    a = Node(add, (1, 2))
    b = Node(add, (1, 2))
    root = Node(add, (a, b))

    lines = root.signature.strip().splitlines()
    assert lines == [
        "n0 = add(x=1, y=2)",
        "add(x=n0, y=n0)",
    ]
