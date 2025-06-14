from pathlib import Path
from contextlib import nullcontext

import yaml  # type: ignore[import]
import pytest
import joblib  # type: ignore[import]
from node.node import Node, Config, ChainCache, MemoryLRU, DiskJoblib


def test_flow_example(flow_factory):
    flow = flow_factory()

    @flow.node()
    def add(x, y):
        return x + y

    @flow.node()
    def square(z):
        return z * z

    root = square(add(2, 3))
    assert flow.run(root) == 25


def test_node_get(flow_factory):
    flow = flow_factory()

    @flow.node()
    def add(x, y):
        return x + y

    node = add(2, 3)
    assert node.get() == 5
    # Second call should reuse cache
    assert node.get() == 5


def test_generate_populates_cache(flow_factory):
    flow = flow_factory()
    calls = []

    @flow.node()
    def inc(x):
        calls.append(x)
        return x + 1

    node = inc(2)
    flow.generate(node)
    assert calls == [2]
    # get should reuse cache without recomputing
    assert node.get() == 3
    assert calls == [2]


def test_cache_skips_execution(flow_factory):
    flow = flow_factory()
    calls = []

    @flow.node()
    def double(x):
        calls.append(x)
        return x * 2

    node = double(4)
    assert flow.run(node) == 8
    assert calls == [4]

    # Second run should come entirely from cache
    assert flow.run(node) == 8
    assert calls == [4]


def test_defaults_override(flow_factory):
    conf = Config({"add": {"y": 5}})
    flow = flow_factory(config=conf)

    @flow.node()
    def add(x, y=1):
        return x + y

    node = add(3)
    assert flow.run(node) == 8


def test_positional_args_ignore_config(flow_factory):
    conf = Config({"add": {"y": 5}})
    flow = flow_factory(config=conf)

    @flow.node()
    def add(x, y):
        return x + y

    node = add(2, 3)
    assert flow.run(node) == 5


def test_config_from_yaml(flow_factory):
    cfg_path = Path(__file__).with_name("config.yaml")
    with open(cfg_path) as f:
        defaults = yaml.safe_load(f)

    flow = flow_factory(config=Config(defaults))

    @flow.node()
    def add(x, y=1):
        return x + y

    node = add(3)
    assert flow.run(node) == 8


def test_build_script_repr(flow_factory):
    flow = flow_factory()

    @flow.node()
    def add(x, y):
        return x + y

    @flow.node()
    def square(z):
        return z * z

    node = square(add(2, 3))
    script = repr(node)
    assert script.strip() == "square(z=add(x=2, y=3))"


def test_linear_chain_repr(flow_factory):
    flow = flow_factory()

    @flow.node()
    def f1(a):
        return a

    @flow.node()
    def f2(a):
        return a

    @flow.node()
    def f3(a):
        return a

    node = f1(f2(f3(1)))
    assert repr(node).strip() == "f1(a=f2(a=f3(a=1)))"


def test_diamond_dependency(flow_factory):
    flow = flow_factory()
    calls = []

    @flow.node()
    def base(x):
        calls.append("base")
        return x + 1

    @flow.node()
    def left(a):
        calls.append("left")
        return a * 2

    @flow.node()
    def right(a):
        calls.append("right")
        return a + 3

    @flow.node()
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


def test_node_deduplication(flow_factory):
    flow = flow_factory()

    @flow.node()
    def add(x, y):
        return x + y

    n1 = add(1, 2)
    n2 = add(1, 2)
    assert n1 is n2


def test_repr_shared_nodes(flow_factory):
    """repr should reuse the same variable for identical nodes."""
    flow = flow_factory()

    @flow.node()
    def add(x, y):
        return x + y

    @flow.node()
    def combine(a, b):
        return a + b

    node = combine(add(1, 2), add(1, 2))
    script = repr(node).strip().splitlines()
    assert script == [
        "n0 = add(x=1, y=2)",
        "combine(a=n0, b=n0)",
    ]


def test_set_canonicalization(flow_factory):
    flow = flow_factory()
    calls = []

    @flow.node()
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


def test_chaincache_promotion(flow_factory, tmp_path):
    mem = MemoryLRU()
    disk = DiskJoblib(tmp_path)
    flow = flow_factory(cache=ChainCache([mem, disk]))

    @flow.node()
    def add(x, y):
        return x + y

    node = add(2, 3)
    flow.run(node)
    assert node.signature in mem._lru

    mem._lru.clear()
    assert node.signature not in mem._lru

    flow.run(node)
    assert node.signature in mem._lru


def test_parallel_execution(flow_factory):
    flow = flow_factory(executor="thread", workers=2)

    @flow.node()
    def slow(v):
        import time

        time.sleep(0.2)
        return v

    @flow.node()
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


def test_dict_canonicalization(flow_factory):
    flow = flow_factory()

    @flow.node()
    def ident(x):
        return x

    n1 = ident({"a": 1, "b": 2})
    n2 = ident({"b": 2, "a": 1})

    assert n1.signature == n2.signature
    assert n1 is n2


def test_callbacks_invoked(flow_factory):
    events = []

    def on_node_end(node, dur, cached):
        events.append(("node", cached))

    def on_flow_end(root, wall, count):
        events.append(("flow", count))

    flow = flow_factory()
    flow.engine.on_node_end = on_node_end
    flow.engine.on_flow_end = on_flow_end

    @flow.node()
    def add(x, y):
        return x + y

    node = add(1, 2)
    assert flow.run(node) == 3
    assert events == [("node", False), ("flow", 1)]

    events.clear()
    assert flow.run(node) == 3
    assert events == [("node", True), ("flow", 0)]


def test_cache_scripts(flow_factory, tmp_path):
    disk = DiskJoblib(tmp_path)
    flow = flow_factory(cache=ChainCache([MemoryLRU(), disk]))

    @flow.node()
    def base(x):
        return x + 1

    @flow.node()
    def left(a):
        return a * 2

    @flow.node()
    def right(a):
        return a + 3

    @flow.node()
    def final(x, y):
        return x + y

    b = base(4)
    root = final(left(b), right(b))
    expected = (4 + 1) * 2 + (4 + 1) + 3
    assert flow.run(root) == expected

    pkls = sorted(tmp_path.rglob("*.pkl"))
    pys = sorted(tmp_path.rglob("*.py"))
    assert len(pkls) == 4

    for p in pkls:
        if len(p.stem) == 32:
            py = p.with_suffix(".py")
            assert py.exists()
        else:
            assert not p.with_suffix(".py").exists()
    assert len(pys) == sum(len(p.stem) == 32 for p in pkls)


def test_cache_fallback_hash(flow_factory, tmp_path, monkeypatch):
    disk = DiskJoblib(tmp_path)
    flow = flow_factory(cache=ChainCache([MemoryLRU(), disk]))

    @flow.node()
    def inc(x):
        return x + 1

    orig_dump = joblib.dump

    def bad_first_dump(obj, path, *args, **kwargs):
        if not hasattr(bad_first_dump, "done"):
            bad_first_dump.done = True
            raise OSError("fail")
        return orig_dump(obj, path, *args, **kwargs)

    monkeypatch.setattr(joblib, "dump", bad_first_dump)

    node = inc(5)
    assert flow.run(node) == 6

    expr_file = disk._expr_path(node.signature)
    assert not expr_file.exists()

    pkl = disk._hash_path(node.signature)
    assert pkl.exists()
    assert pkl.with_suffix(".py").exists()


def test_ignore_signature_fields(flow_factory):
    flow = flow_factory()

    @flow.node(ignore=["large_df", "model"])
    def add(x, y, large_df=None, model=None):
        return x + y

    n1 = add(1, 2, large_df=[1, 2], model="a")
    n2 = add(1, 2, large_df=[3, 4], model="b")

    assert n1 is n2
    assert n1.signature == "add(x=1, y=2)"
    assert flow.run(n1) == 3


def test_delete_cache(flow_factory, tmp_path):
    mem = MemoryLRU()
    disk = DiskJoblib(tmp_path)
    flow = flow_factory(cache=ChainCache([mem, disk]))
    calls = []

    @flow.node()
    def add(x, y):
        calls.append(1)
        return x + y

    node = add(1, 2)
    assert flow.run(node) == 3
    assert node.signature in mem._lru

    p = disk._expr_path(node.signature)
    if not p.exists():
        p = disk._hash_path(node.signature)
    assert p.exists()

    node.delete_cache()

    assert node.signature not in mem._lru
    assert not p.exists()

    assert flow.run(node) == 3
    assert calls == [1, 1]


def test_default_reporter(flow_factory):
    class DummyReporter:
        def __init__(self):
            self.count = 0

        def attach(self, engine, root):
            self.count += 1
            return nullcontext()

    reporter = DummyReporter()
    flow = flow_factory(reporter=reporter)

    @flow.node()
    def add(x, y):
        return x + y

    node = add(1, 2)
    assert flow.run(node) == 3
    assert reporter.count == 1

    node.delete_cache()
    extra = DummyReporter()
    assert flow.run(node, reporter=extra) == 3
    assert reporter.count == 1
    assert extra.count == 1
