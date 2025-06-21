from node import Flow
from node.node import ChainCache, DiskJoblib, MemoryLRU


def test_process_pickling_fallback(tmp_path):
    disk = DiskJoblib(tmp_path)
    orig_put = disk.put

    def safe_put(key, value):
        try:
            orig_put(key, value)
        except Exception:
            pass

    disk.put = safe_put  # type: ignore[assignment]
    flow = Flow(cache=ChainCache([MemoryLRU(), disk]), executor="process")

    @flow.node()
    def make_fn(x):
        def inner() -> int:
            return x

        return inner

    node = make_fn(5)
    fn = flow.run(node)
    assert fn() == 5
    p = disk._path(node.key, ".py")
    assert p.exists()
