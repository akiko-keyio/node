import pytest
from node.node import Node, Flow, Config, ChainCache, MemoryLRU, DiskJoblib


def test_node_get_without_flow():
    node = Node(lambda: 1)
    with pytest.raises(RuntimeError):
        node.get()


def test_node_generate_without_flow():
    node = Node(lambda: 1)
    with pytest.raises(RuntimeError):
        node.generate()


def test_node_delete_cache_without_flow():
    node = Node(lambda: 1)
    with pytest.raises(RuntimeError):
        node.delete_cache()


def test_save_script_skipped_when_expr_ok(tmp_path):
    disk = DiskJoblib(tmp_path)
    flow = Flow(cache=ChainCache([MemoryLRU(), disk]), log=False)

    @flow.node()
    def echo(x):
        return x

    node = echo(1)
    flow.run(node)
    assert disk._expr_path(node.signature).exists()
    assert not disk._hash_path(node.signature, ".py").exists()


def test_nested_structure_canonicalization(tmp_path):
    flow = Flow(cache=ChainCache([MemoryLRU(), DiskJoblib(tmp_path)]), log=False)

    @flow.node()
    def ident(x):
        return x

    d1 = {"a": {1, 2}, "b": {"x": 1}}
    d2 = {"b": {"x": 1}, "a": {2, 1}}
    n1 = ident(d1)
    n2 = ident(d2)
    assert n1.signature == n2.signature
    assert n1 is n2


def test_node_in_set_canonicalization(tmp_path):
    flow = Flow(cache=ChainCache([MemoryLRU(), DiskJoblib(tmp_path)]), log=False)

    @flow.node()
    def inc(x):
        return x + 1

    n1 = inc(1)
    n2 = inc(1)

    @flow.node()
    def wrap(s):
        return len(s)

    w1 = wrap({n1})
    w2 = wrap({n2})
    assert w1.signature == w2.signature
    assert w1 is w2


def test_flow_task_alias(tmp_path):
    flow = Flow(cache=ChainCache([MemoryLRU(), DiskJoblib(tmp_path)]), log=False)

    @flow.task()
    def inc(x):
        return x + 1

    node = inc(2)
    assert flow.run(node) == 3


def test_defaults_nested(tmp_path):
    cfg = Config({"add": {"y": 5}})
    flow = Flow(
        config=cfg,
        cache=ChainCache([MemoryLRU(), DiskJoblib(tmp_path)]),
        log=False,
    )

    @flow.node()
    def add(x, y=1):
        return x + y

    @flow.node()
    def wrap(z):
        return z + 1

    root = wrap(add(2))
    assert flow.run(root) == 8


def test_run_none_reporter(tmp_path):
    flow = Flow(cache=ChainCache([MemoryLRU(), DiskJoblib(tmp_path)]), log=False)

    @flow.node()
    def add(x, y):
        return x + y

    node = add(1, 2)
    assert flow.run(node, reporter=None) == 3


def test_ignore_nested_field(tmp_path):
    flow = Flow(cache=ChainCache([MemoryLRU(), DiskJoblib(tmp_path)]), log=False)

    @flow.node(ignore=["meta"])
    def add(x, y, meta=None):
        return x + y

    n1 = add(1, 2, meta={"a": 1})
    n2 = add(1, 2, meta={"b": 2})
    assert n1 is n2


def test_chaincache_delete(tmp_path):
    mem = MemoryLRU()
    disk = DiskJoblib(tmp_path)
    chain = ChainCache([mem, disk])

    chain.put("k", 1)
    assert "k" in mem._lru
    chain.delete("k")
    assert "k" not in mem._lru
    assert not disk._expr_path("k").exists()


def test_on_flow_end_callback(tmp_path):
    events = []

    def on_flow_end(root, wall, count):
        events.append(count)

    flow = Flow(cache=ChainCache([MemoryLRU(), DiskJoblib(tmp_path)]), log=False)
    flow.engine.on_flow_end = on_flow_end

    @flow.node()
    def add(x, y):
        return x + y

    node = add(1, 2)
    assert flow.run(node) == 3
    assert events == [1]


def test_varargs_kwargs(tmp_path):
    flow = Flow(cache=ChainCache([MemoryLRU(), DiskJoblib(tmp_path)]), log=False)

    @flow.node()
    def foo(*args, **kwargs):
        return sum(args) + sum(kwargs.values())

    node = foo(1, 2, a=3)
    assert flow.run(node) == 6


def test_none_default(tmp_path):
    flow = Flow(cache=ChainCache([MemoryLRU(), DiskJoblib(tmp_path)]), log=False)

    @flow.node()
    def foo(x=None):
        return x

    n1 = foo()
    n2 = foo(None)
    assert n1 is n2


def test_bool_signature(tmp_path):
    flow = Flow(cache=ChainCache([MemoryLRU(), DiskJoblib(tmp_path)]), log=False)

    @flow.node()
    def ident(x):
        return x

    n1 = ident(True)
    n2 = ident(True)
    assert n1 is n2
    assert n1.signature == "ident(x=True)"


def test_different_functions_do_not_clash(tmp_path):
    flow = Flow(cache=ChainCache([MemoryLRU(), DiskJoblib(tmp_path)]), log=False)

    @flow.node()
    def f(x):
        return x

    @flow.node()
    def g(x):
        return x

    assert f(1) is not g(1)


def test_config_defaults_unknown():
    cfg = Config({"foo": {"x": 1}})
    assert cfg.defaults("bar") == {}


def test_ignore_node_parameter(tmp_path):
    flow = Flow(cache=ChainCache([MemoryLRU(), DiskJoblib(tmp_path)]), log=False)

    @flow.node()
    def inc(x):
        return x + 1

    @flow.node(ignore=["extra"])
    def add(x, y, extra=None):
        return x + y

    n1 = add(1, 2, extra=inc(10))
    n2 = add(1, 2, extra=inc(10))
    assert n1 is n2


def test_chaincache_delete_multiple(tmp_path):
    mem1 = MemoryLRU()
    mem2 = MemoryLRU()
    disk = DiskJoblib(tmp_path)
    chain = ChainCache([mem1, mem2, disk])
    chain.put("k", 1)
    assert chain.get("k")[0]
    chain.delete("k")
    for c in (mem1, mem2):
        assert "k" not in c._lru
    assert not disk._expr_path("k").exists()


def test_generate_then_run_uses_cache(tmp_path):
    flow = Flow(cache=ChainCache([MemoryLRU(), DiskJoblib(tmp_path)]), log=False)
    calls = []

    @flow.node()
    def inc(x):
        calls.append(x)
        return x + 1

    node = inc(5)
    node.generate()
    assert calls == [5]
    assert flow.run(node) == 6
    assert calls == [5]
