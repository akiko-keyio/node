import node
from node import Node, ChainCache, MemoryLRU, DiskJoblib


def test_repr_matches_script(runtime_factory, tmp_path):
    rt = runtime_factory(cache=ChainCache([MemoryLRU(), DiskJoblib(tmp_path)]))

    @node.define()
    def add(x, y):
        return x + y

    @node.define()
    def square(z):
        return z * z

    linear = square(add(2, 3))
    assert repr(linear) == linear.script

    diamond = add(add(1, 2), add(1, 2))
    assert repr(diamond) == diamond.script


def test_branch_no_diamond(runtime_factory, tmp_path):
    """Branching without shared nodes should not expand to a script."""
    rt = runtime_factory(cache=ChainCache([MemoryLRU(), DiskJoblib(tmp_path)]))

    @node.define()
    def add(x, y):
        return x + y

    @node.define()
    def square(z):
        return z * z

    n = square(add(square(1), square(2)))
    # script format: header + body lines with simplified variable names
    lines = n.script.strip().splitlines()
    # Header is "# hash = ..."
    assert lines[0].startswith("# hash = ")
    body_lines = lines[1:]
    # With simplified naming: square_0, square_1, add_0, square_2
    assert len(body_lines) == 4
    assert "square_0 = square(z=1)" in body_lines[0]
    assert "square_1 = square(z=2)" in body_lines[1]
    assert "add_0 = add(x=square_0, y=square_1)" in body_lines[2]
    assert "square_2 = square(z=add_0)" in body_lines[3]


def test_script_key_canonicalization():
    def identity(x=None, **kw):
        return (x, kw)

    # 使用 inputs 字典格式
    n1 = Node(identity, {"x": {2, 1}})
    n2 = Node(identity, {"x": {1, 2}})

    assert n1.script == n2.script

    d1 = Node(identity, {"d": {"b": 2, "a": 1}})
    d2 = Node(identity, {"d": {"a": 1, "b": 2}})

    assert d1.script == d2.script


def test_script_dedup():
    def add(x, y):
        return x + y

    # 使用 inputs 字典格式
    a = Node(add, {"x": 1, "y": 2})
    b = Node(add, {"x": 1, "y": 2})
    root = Node(add, {"x": a, "y": b})

    lines = root.script.strip().splitlines()
    # Header + 2 body lines (deduped since a == b)
    assert lines[0].startswith("# hash = ")
    body_lines = lines[1:]
    assert len(body_lines) == 2
    assert "add_0 = add(x=1, y=2)" in body_lines[0]
    assert "add_1 = add(x=add_0, y=add_0)" in body_lines[1]

