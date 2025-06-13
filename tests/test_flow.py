from node.node import Node, Flow, Config, ChainCache, MemoryLRU, DiskJoblib
import pytest


def test_flow_example(tmp_path):
    flow = Flow(cache=ChainCache([MemoryLRU(), DiskJoblib(tmp_path)]), log=False)

    @flow.task()
    def add(x, y):
        return x + y

    @flow.task()
    def square(z):
        return z * z

    root = square(add(2, 3))
    assert flow.run(root) == 25


def test_cache_skips_execution(tmp_path):
    flow = Flow(cache=ChainCache([MemoryLRU(), DiskJoblib(tmp_path)]), log=False)
    calls = []

    @flow.task()
    def double(x):
        calls.append(x)
        return x * 2

    node = double(4)
    assert flow.run(node) == 8
    assert calls == [4]

    # Second run should come entirely from cache
    assert flow.run(node) == 8
    assert calls == [4]


def test_defaults_override(tmp_path):
    conf = Config({"add": {"y": 5}})
    flow = Flow(config=conf, cache=ChainCache([MemoryLRU(), DiskJoblib(tmp_path)]), log=False)

    @flow.task()
    def add(x, y=1):
        return x + y

    node = add(3)
    assert flow.run(node) == 8


def test_build_script_repr(tmp_path):
    flow = Flow(cache=ChainCache([MemoryLRU(), DiskJoblib(tmp_path)]), log=False)

    @flow.task()
    def add(x, y):
        return x + y

    @flow.task()
    def square(z):
        return z * z

    node = square(add(2, 3))
    script = repr(node)
    assert script.strip().splitlines() == [
        "n0 = add(2, 3)",
        "n1 = square(n0)",
        "n1",
    ]

def test_diamond_dependency(tmp_path):
    flow = Flow(cache=ChainCache([MemoryLRU(), DiskJoblib(tmp_path)]), log=False)
    calls = []

    @flow.task()
    def base(x):
        calls.append("base")
        return x + 1

    @flow.task()
    def left(a):
        calls.append("left")
        return a * 2

    @flow.task()
    def right(a):
        calls.append("right")
        return a + 3

    @flow.task()
    def final(x, y):
        calls.append("final")
        return x + y

    b = base(4)
    root = final(left(b), right(b))
    expected = (4 + 1) * 2 + (4 + 1) + 3
    assert flow.run(root) == expected
    assert calls.count("base") == 1
    assert calls.count("left") == 1
    assert calls.count("right") == 1
    assert calls.count("final") == 1

    # running again should hit the cache only
    assert flow.run(root) == expected
    assert calls.count("base") == 1
    assert calls.count("left") == 1
    assert calls.count("right") == 1
    assert calls.count("final") == 1


def test_node_deduplication(tmp_path):
    flow = Flow(cache=ChainCache([MemoryLRU(), DiskJoblib(tmp_path)]), log=False)

    @flow.task()
    def add(x, y):
        return x + y

    n1 = add(1, 2)
    n2 = add(1, 2)
    assert n1 is n2


def test_repr_shared_nodes(tmp_path):
    """repr should reuse the same variable for identical nodes."""
    flow = Flow(cache=ChainCache([MemoryLRU(), DiskJoblib(tmp_path)]), log=False)

    @flow.task()
    def add(x, y):
        return x + y

    @flow.task()
    def combine(a, b):
        return a + b

    node = combine(add(1, 2), add(1, 2))
    script = repr(node).strip().splitlines()
    assert script == [
        "n0 = add(1, 2)",
        "n1 = combine(n0, n0)",
        "n1",
    ]


def test_set_canonicalization(tmp_path):
    flow = Flow(cache=ChainCache([MemoryLRU(), DiskJoblib(tmp_path)]), log=False)
    calls = []

    @flow.task()
    def identity(x):
        calls.append(1)
        return x

    n1 = identity({1, 2})
    n2 = identity({2, 1})

    assert n1.signature == n2.signature
    assert n1 is n2

    flow.run(n1)
    flow.run(n2)
    assert len(calls) == 1


def test_chaincache_promotion(tmp_path):
    mem = MemoryLRU()
    disk = DiskJoblib(tmp_path)
    flow = Flow(cache=ChainCache([mem, disk]), log=False)

    @flow.task()
    def add(x, y):
        return x + y

    node = add(2, 3)
    flow.run(node)
    assert node.signature in mem._lru

    mem._lru.clear()
    assert node.signature not in mem._lru

    flow.run(node)
    assert node.signature in mem._lru


def test_parallel_execution(tmp_path):
    flow = Flow(cache=ChainCache([MemoryLRU(), DiskJoblib(tmp_path)]), executor="thread", workers=2, log=False)

    @flow.task()
    def slow(v):
        import time
        time.sleep(0.2)
        return v

    @flow.task()
    def combine(a, b):
        return a + b

    root = combine(slow(1), slow(2))
    import time
    t0 = time.perf_counter()
    assert flow.run(root) == 3
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.4


def test_cycle_detection():
    """Creating a node that depends on itself should raise."""
    def ident(x):
        return x

    node = Node.__new__(Node)
    with pytest.raises(ValueError):
        Node.__init__(node, ident, (node,), {})


def test_dict_canonicalization(tmp_path):
    flow = Flow(cache=ChainCache([MemoryLRU(), DiskJoblib(tmp_path)]), log=False)

    @flow.task()
    def ident(x):
        return x

    n1 = ident({"a": 1, "b": 2})
    n2 = ident({"b": 2, "a": 1})

    assert n1.signature == n2.signature
    assert n1 is n2


def test_callbacks_invoked(tmp_path):
    events = []

    def on_node_end(node, dur, cached):
        events.append(("node", cached))

    def on_flow_end(root, wall, count):
        events.append(("flow", count))

    flow = Flow(
        cache=ChainCache([MemoryLRU(), DiskJoblib(tmp_path)]),
        log=False,
    )
    flow.engine.on_node_end = on_node_end
    flow.engine.on_flow_end = on_flow_end

    @flow.task()
    def add(x, y):
        return x + y

    node = add(1, 2)
    assert flow.run(node) == 3
    assert events == [("node", False), ("flow", 1)]

    events.clear()
    assert flow.run(node) == 3
    assert events == [("node", True), ("flow", 0)]
