import node
from node import ChainCache, DiskJoblib, MemoryLRU


def test_process_pickling_fallback(tmp_path):
    disk = DiskJoblib(tmp_path)
    orig_put = disk.put

    def safe_put(fn_name, hash_value, value):
        try:
            orig_put(fn_name, hash_value, value)
        except Exception:
            pass

    disk.put = safe_put  # type: ignore[assignment]
    node.reset()
    rt = node.configure(
        cache=ChainCache([MemoryLRU(), disk]),
        executor="process",
        continue_on_error=False,
        validate=False,
    )

    @node.define()
    def make_fn(x):
        def inner() -> int:
            return x

        return inner

    task_node = make_fn(5)
    fn = rt.run(task_node)
    assert fn() == 5
    p = disk._path(task_node.fn.__name__, task_node._hash, ".py")
    assert p.exists()
