import cProfile
import io
import pstats
import time

import node
from node import ChainCache, DiskJoblib, MemoryLRU


def _profile_run(tmp_path, small_file: int) -> tuple[float, str]:
    cache = ChainCache([MemoryLRU(), DiskJoblib(tmp_path, small_file=small_file)])
    node.reset()
    rt = node.configure(cache=cache, executor="thread", workers=8, reporter=None)
    rt.reporter = None

    @node.define()
    def text(i: int) -> str:
        return f"text-{i}"

    # root = gather([text(i) for i in range(500)])
    # Replacement manual gathering
    @node.define()
    def gather_manual(*args): return list(args)
    root = gather_manual(*[text(i) for i in range(500)])
    rt.run(root)
    cache.caches[0]._lru.clear()

    prof = cProfile.Profile()
    prof.enable()
    start = time.perf_counter()
    rt.run(root)
    duration = time.perf_counter() - start
    prof.disable()

    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).sort_stats("cumtime").print_stats(5)
    return duration, buf.getvalue()


def test_diskjoblib_small_file_performance(tmp_path):
    dur_joblib, stat_joblib = _profile_run(tmp_path / "joblib", 0)
    dur_pickle, stat_pickle = _profile_run(tmp_path / "pickle", 1_000_000)
    assert "joblib" in stat_joblib
    assert "joblib" not in stat_pickle
    assert dur_pickle < dur_joblib
