from node import ChainCache, DiskCache, MemoryLRU
from node.core import build_graph
import node


def test_build_graph_uses_lightweight_hit_check_not_get(
    tmp_path, monkeypatch, runtime_factory
):
    """构图阶段只判断命中，不应触发磁盘反序列化。"""
    mem = MemoryLRU()
    disk = DiskCache(tmp_path)
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
    """轻量命中检查命中但缓存损坏时，执行阶段应自动回退重算依赖。"""
    calls: list[str] = []
    disk = DiskCache(tmp_path)
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
    # 仅写入一个损坏的 root 缓存文件：命中检查为 True，但 get 会失败并触发删除。
    corrupt_path = disk._path(root.fn.__name__, root._hash)
    corrupt_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt_path.write_bytes(b"corrupt")

    assert rt.run(root) == 8
    assert calls == ["base", "twice"]


def test_disk_cache_has_entry_uses_lazy_namespace_index(tmp_path, monkeypatch):
    disk = DiskCache(tmp_path)
    fn_name = "demo"
    hash_value = int("deadbeef", 16)
    disk.put(fn_name, hash_value, {"ok": True})

    assert disk._has_entry(fn_name, hash_value) is True

    def _unexpected_path(*args, **kwargs):
        raise AssertionError("indexed _has_entry should not rebuild file path")

    monkeypatch.setattr(disk, "_path", _unexpected_path)
    assert disk._has_entry(fn_name, hash_value) is True


def test_disk_cache_lazy_index_tracks_put_and_delete(tmp_path):
    disk = DiskCache(tmp_path)
    fn_name = "demo"
    hash_value = int("abc123", 16)

    assert disk._has_entry(fn_name, hash_value) is False
    disk.put(fn_name, hash_value, 1)
    assert disk._has_entry(fn_name, hash_value) is True
    disk.delete(fn_name, hash_value)
    assert disk._has_entry(fn_name, hash_value) is False


def test_disk_cache_lazy_index_handles_nested_namespace(tmp_path):
    disk = DiskCache(tmp_path)
    fn_name = "demo/nested"
    hash_value = int("f00baa", 16)

    disk.put(fn_name, hash_value, 1)
    assert disk._has_entry(fn_name, hash_value) is True

    p = disk._path(fn_name, hash_value)
    p.unlink()
    disk._handle_corrupt_cache(p, ValueError("boom"))
    assert disk._has_entry(fn_name, hash_value) is False


