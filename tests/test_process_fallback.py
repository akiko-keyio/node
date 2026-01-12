from node import Runtime
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
    rt = Runtime(
        cache=ChainCache([MemoryLRU(), disk]),
        executor="process",
        continue_on_error=False,
        validate=False,
    )

    @rt.define()
    def make_fn(x):
        def inner() -> int:
            return x

        return inner

    node = make_fn(5)
    fn = rt.run(node)
    assert fn() == 5
    p = disk._path(node.fn.__name__, node._hash, ".py")
    assert p.exists()
