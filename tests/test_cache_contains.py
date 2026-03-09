from node import ChainCache, DiskJoblib, MemoryLRU
from node.core import build_graph
import node


def test_build_graph_uses_contains_not_get(tmp_path, monkeypatch, runtime_factory):
    """构图阶段只判断命中，不应触发磁盘反序列化。"""
    mem = MemoryLRU()
    disk = DiskJoblib(tmp_path)
    cache = ChainCache([mem, disk])
    runtime_factory(cache=cache)

    @node.define()
    def inc(x):
        return x + 1

    root = inc(1)
    disk.put(root.fn.__name__, root._hash, 2)

    def _unexpected_get(*args, **kwargs):
        raise AssertionError("build_graph should not call cache.get")

    monkeypatch.setattr(disk, "get", _unexpected_get)

    order, edges = build_graph(root, cache)
    assert order == [root]
    assert edges[root] == []


def test_corrupt_cached_hit_recovers_with_dependencies(tmp_path, runtime_factory):
    """contains 命中但缓存损坏时，执行阶段应自动回退重算依赖。"""
    calls: list[str] = []
    disk = DiskJoblib(tmp_path)
    rt = runtime_factory(cache=ChainCache([MemoryLRU(), disk]))

    @node.define()
    def base(x):
        calls.append("base")
        return x + 1

    @node.define()
    def twice(y):
        calls.append("twice")
        return y * 2

    root = twice(base(3))
    # 仅写入一个损坏的 root 缓存文件：contains=True，但 get 会失败并触发删除。
    disk._path(root.fn.__name__, root._hash).write_bytes(b"corrupt")

    assert rt.run(root) == 8
    assert calls == ["base", "twice"]
