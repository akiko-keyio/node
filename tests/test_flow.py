from pathlib import Path
from contextlib import nullcontext
import threading

import node
from node import Node, Config, ChainCache, MemoryLRU, DiskJoblib
import pytest


def test_flow_example(runtime_factory):
    rt = runtime_factory()

    @node.define()
    def add(x, y):
        return x + y

    @node.define()
    def square(z):
        return z * z

    root = square(add(2, 3))
    assert rt.run(root) == 25


def test_node_get(runtime_factory):
    rt = runtime_factory()

    @node.define()
    def add(x, y):
        return x + y

    n = add(2, 3)
    assert n() == 5
    # Second call should reuse cache
    assert n() == 5



def test_cache_skips_execution(runtime_factory):
    rt = runtime_factory()
    calls = []

    @node.define()
    def double(x):
        calls.append(x)
        return x * 2

    n = double(4)
    assert rt.run(n) == 8
    assert calls == [4]

    # Second run should come entirely from cache
    assert rt.run(n) == 8
    assert calls == [4]


def test_create_overwrites_cache(runtime_factory):
    rt = runtime_factory()
    calls = []

    @node.define()
    def add(x):
        calls.append(x)
        return x + 1

    n = add(1)
    assert rt.run(n) == 2
    assert calls == [1]
    # cache hit
    assert n() == 2
    assert calls == [1]

    assert n(force=True) == 2
    assert calls == [1, 1]
    # subsequent run uses cache again
    assert n() == 2
    assert calls == [1, 1]


def test_get_no_cache(runtime_factory):
    rt = runtime_factory()
    calls = []

    @node.define(cache=False)
    def inc(x):
        calls.append(x)
        return x + 1

    n = inc(2)

    assert n() == 3
    assert calls == [2]
    assert n() == 3
    assert calls == [2, 2]

    assert n() == 3
    assert calls == [2, 2, 2]


def test_defaults_override(runtime_factory):
    conf = Config({"add": {"y": 5}})
    rt = runtime_factory(config=conf)

    @node.define()
    def add(x, y=1):
        return x + y

    n = add(3)
    assert rt.run(n) == 8


def test_positional_args_ignore_config(runtime_factory):
    conf = Config({"add": {"y": 5}})
    rt = runtime_factory(config=conf)

    @node.define()
    def add(x, y):
        return x + y

    n = add(2, 3)
    assert rt.run(n) == 5


def test_config_from_yaml(runtime_factory):
    cfg_path = Path(__file__).with_name("config.yaml")
    rt = runtime_factory(config=Config(cfg_path))

    @node.define()
    def add(x, y=1):
        return x + y

    n = add(3)
    assert rt.run(n) == 8


def test_build_script_repr(runtime_factory):
    rt = runtime_factory()

    @node.define()
    def add(x, y):
        return x + y

    @node.define()
    def square(z):
        return z * z

    n = square(add(2, 3))
    lines = repr(n).strip().splitlines()
    # New format: header + simplified variable names
    assert lines[0].startswith("# hash = ")
    assert "add_0 = add(x=2, y=3)" in lines[1]
    assert "square_0 = square(z=add_0)" in lines[2]


def test_linear_chain_repr(runtime_factory):
    rt = runtime_factory()

    @node.define()
    def f1(a):
        return a

    @node.define()
    def f2(a):
        return a

    @node.define()
    def f3(a):
        return a

    n = f1(f2(f3(1)))
    lines = repr(n).strip().splitlines()
    # New format: header + simplified variable names
    assert lines[0].startswith("# hash = ")
    assert "f3_0 = f3(a=1)" in lines[1]
    assert "f2_0 = f2(a=f3_0)" in lines[2]
    assert "f1_0 = f1(a=f2_0)" in lines[3]


def test_diamond_dependency(runtime_factory):
    rt = runtime_factory()
    calls = []

    @node.define()
    def base(x):
        calls.append("base")
        return x + 1

    @node.define()
    def left(a):
        calls.append("left")
        return a * 2

    @node.define()
    def right(a):
        calls.append("right")
        return a + 3

    @node.define()
    def final(x, y):
        calls.append("final")
        return x + y

    b = base(4)
    root = final(left(b), right(b))
    expected = (4 + 1) * 2 + (4 + 1) + 3
    assert rt.run(root) == expected
    assert calls.count("base") == 1
    assert calls.count("left") == 1
    assert calls.count("right") == 1
    assert calls.count("final") == 1

    # running again should hit the cache only
    assert rt.run(root) == expected
    assert calls.count("base") == 1
    assert calls.count("left") == 1
    assert calls.count("right") == 1
    assert calls.count("final") == 1


def test_node_deduplication(runtime_factory):
    rt = runtime_factory()

    @node.define()
    def add(x, y):
        return x + y

    n1 = add(1, 2)
    n2 = add(1, 2)
    assert n1 is n2


def test_repr_shared_nodes(runtime_factory):
    """repr should reuse the same variable for identical nodes."""
    rt = runtime_factory()

    @node.define()
    def add(x, y):
        return x + y

    @node.define()
    def combine(a, b):
        return a + b

    n = combine(add(1, 2), add(1, 2))
    lines = repr(n).strip().splitlines()
    # New format: header + simplified variable names (deduped)
    assert lines[0].startswith("# hash = ")
    assert "add_0 = add(x=1, y=2)" in lines[1]
    assert "combine_0 = combine(a=add_0, b=add_0)" in lines[2]


def test_set_canonicalization(runtime_factory):
    rt = runtime_factory()
    calls = []

    @node.define()
    def identity(x):
        calls.append(1)
        return x

    n1 = identity({1, 2})
    n2 = identity({2, 1})

    assert n1.script == n2.script
    assert n1 is n2

    rt.run(n1)
    rt.run(n2)
    assert len(calls) == 1


def test_chaincache_promotion(runtime_factory, tmp_path):
    mem = MemoryLRU()
    disk = DiskJoblib(tmp_path)
    rt = runtime_factory(cache=ChainCache([mem, disk]))

    @node.define()
    def add(x, y):
        return x + y

    n = add(2, 3)
    rt.run(n)
    assert n._hash in mem._lru

    mem._lru.clear()
    assert n._hash not in mem._lru

    rt.run(n)
    assert n._hash in mem._lru


def test_parallel_execution(runtime_factory):
    rt = runtime_factory(executor="thread", workers=2)

    @node.define()
    def slow(v):
        import time

        time.sleep(0.2)
        return v

    @node.define()
    def combine(a, b):
        return a + b

    root = combine(slow(1), slow(2))
    import time

    t0 = time.perf_counter()
    assert rt.run(root) == 3
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.4


def test_node_worker_limit(runtime_factory):
    rt = runtime_factory(executor="thread", workers=4)

    @node.define(workers=1)
    def slow(v):
        import time

        time.sleep(0.2)
        return v

    @node.define()
    def combine(a, b):
        return a + b

    root = combine(slow(1), slow(2))
    import time

    t0 = time.perf_counter()
    assert rt.run(root) == 3
    elapsed = time.perf_counter() - t0
    assert elapsed >= 0.4


@pytest.mark.parametrize("dw,nw", [(1, None), (2, 1)])
def test_no_thread_pool_for_single_worker(runtime_factory, monkeypatch, dw, nw):
    import node.runtime as node_module

    called = False

    def fail_pool(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("ThreadPoolExecutor should not be used")

    monkeypatch.setattr(node_module, "ThreadPoolExecutor", fail_pool)

    rt = runtime_factory(executor="thread", workers=dw)

    @node.define(workers=nw)
    def slow(v):
        return v

    assert rt.run(slow(1)) == 1

    assert not called


def test_local_node_runs_in_main_thread(runtime_factory):
    rt = runtime_factory(executor="thread", workers=2)

    @node.define(local=True)
    def which_thread() -> str:
        import threading

        return threading.current_thread().name

    @node.define()
    def wrapper(v: str) -> tuple[str, str]:
        import threading

        return threading.current_thread().name, v

    name, inner = rt.run(wrapper(which_thread()))
    assert inner == "MainThread"
    assert name != "MainThread"


@pytest.mark.skip(reason="Cycle detection temporarily disabled")
def test_cycle_detection():
    """Creating a node that depends on itself should raise."""

    def ident(x):
        return x

    node = Node.__new__(Node)
    with pytest.raises(ValueError):
        Node.__init__(node, ident, {"x": node})


def test_dict_canonicalization(runtime_factory):
    rt = runtime_factory()

    @node.define()
    def ident(x):
        return x

    n1 = ident({"a": 1, "b": 2})
    n2 = ident({"b": 2, "a": 1})

    assert n1.script == n2.script
    assert n1 is n2


def test_callbacks_invoked(runtime_factory):
    events = []

    def on_node_end(node, dur, cached, failed):
        events.append(("node", cached, failed))

    def on_flow_end(root, wall, count, fails):
        events.append(("rt", count, fails))

    rt = runtime_factory()
    rt.on_node_end = on_node_end
    rt.on_flow_end = on_flow_end

    @node.define()
    def add(x, y):
        return x + y

    n = add(1, 2)
    assert rt.run(n) == 3
    assert events == [("node", False, False), ("rt", 1, 0)]

    events.clear()
    assert rt.run(n) == 3
    assert events == [("node", True, False), ("rt", 0, 0)]


def test_cache_scripts(runtime_factory, tmp_path):
    disk = DiskJoblib(tmp_path)
    rt = runtime_factory(cache=ChainCache([MemoryLRU(), disk]))

    @node.define()
    def base(x):
        return x + 1

    @node.define()
    def left(a):
        return a * 2

    @node.define()
    def right(a):
        return a + 3

    @node.define()
    def final(x, y):
        return x + y

    b = base(4)
    root = final(left(b), right(b))
    expected = (4 + 1) * 2 + (4 + 1) + 3
    assert rt.run(root) == expected

    pkls = sorted(tmp_path.rglob("*.pkl"))
    pys = sorted(tmp_path.rglob("*.py"))
    assert len(pkls) == 4
    assert len(pys) == 4


def test_cache_fallback_hash(runtime_factory, tmp_path, monkeypatch):
    disk = DiskJoblib(tmp_path)
    rt = runtime_factory(cache=ChainCache([MemoryLRU(), disk]))

    @node.define()
    def inc(x):
        return x + 1

    orig_put = disk.put

    def bad_first_put(fn_name, hash_value, value):
        if not hasattr(bad_first_put, "done"):
            bad_first_put.done = True
            raise OSError("fail")
        return orig_put(fn_name, hash_value, value)

    monkeypatch.setattr(disk, "put", bad_first_put)

    n = inc(5)
    with pytest.raises(OSError):
        rt.run(n)


def test_ignore_signature_fields(runtime_factory):
    rt = runtime_factory()

    @node.define(ignore=["large_df", "model"])
    def add(x, y, large_df=None, model=None):
        return x + y

    n1 = add(1, 2, large_df=[1, 2], model="a")
    n2 = add(1, 2, large_df=[3, 4], model="b")

    assert n1 is n2
    assert n1.script.endswith("add(x=1, y=2)")
    assert rt.run(n1) == 3


def test_delete_cache(runtime_factory, tmp_path):
    mem = MemoryLRU()
    disk = DiskJoblib(tmp_path)
    rt = runtime_factory(cache=ChainCache([mem, disk]))
    calls = []

    @node.define()
    def add(x, y):
        calls.append(1)
        return x + y

    n = add(1, 2)
    assert rt.run(n) == 3
    assert n._hash in mem._lru

    p = disk._path(n.fn.__name__, n._hash)
    assert p.exists()

    n.invalidate()

    assert n._hash not in mem._lru
    assert not p.exists()

    assert rt.run(n) == 3
    assert calls == [1, 1]


def test_default_reporter(runtime_factory):
    class DummyReporter:
        def __init__(self):
            self.count = 0

        def attach(self, engine, root):
            self.count += 1
            return nullcontext()

    reporter = DummyReporter()
    rt = runtime_factory(reporter=reporter)

    @node.define()
    def add(x, y):
        return x + y

    n = add(1, 2)
    assert rt.run(n) == 3
    assert reporter.count == 1

    n.invalidate()
    extra = DummyReporter()
    assert rt.run(n, reporter=extra) == 3
    assert reporter.count == 1
    assert extra.count == 1


def test_concurrent_node_construction(runtime_factory):
    rt = runtime_factory()

    @node.define()
    def add(x, y):
        return x + y

    results = []

    def build():
        results.append(add(1, 2))

    threads = [threading.Thread(target=build) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(r is results[0] for r in results)


def test_deep_chain_signature(runtime_factory):
    rt = runtime_factory(cache=MemoryLRU())

    @node.define()
    def inc(x):
        return x + 1

    n = inc(0)
    for _ in range(200):
        n = inc(n)

    assert rt.run(n) == 201


def test_cached_dependency_skips_upstream(runtime_factory):
    class TrackingCache(MemoryLRU):
        def __init__(self):
            super().__init__()
            self.calls: list[int] = []

        def get(self, fn_name: str, hash_value: int):  # type: ignore[override]
            self.calls.append(hash_value)
            return super().get(fn_name, hash_value)

    cache = TrackingCache()
    rt = runtime_factory(cache=cache)

    @node.define()
    def c():
        return 1

    c_node = c()

    @node.define()
    def b(x):
        return x + 1

    b_node = b(c_node)

    @node.define()
    def a(y):
        return y + 1

    root = a(b_node)

    assert rt.run(root) == 3

    cache.delete(root.fn.__name__, root._hash)
    cache.calls.clear()

    assert rt.run(root) == 3
    assert c_node._hash not in cache.calls
    assert b_node._hash in cache.calls

