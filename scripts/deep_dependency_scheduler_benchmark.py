"""调度路径专项基准：上万维度 + 深层依赖。

目标
----
针对 Runtime 并行路径中以下过程做定量分析：
1) 处理已完成任务（future.result / sem.release / ts.done / 释放依赖引用）
2) 发现新可运行节点（ts.get_ready）
3) ready 节点前置动作（_resolve / cache.get / _ensure_deps_ready）
4) 提交到线程池（pool.submit + sem.acquire）

运行示例
--------
uv run python -m scripts.deep_dependency_scheduler_benchmark --dim-size 12000 --depth 10
uv run python -m scripts.deep_dependency_scheduler_benchmark --workers 8
"""

from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import re
import statistics
import time
from dataclasses import dataclass

import node
from node import MemoryLRU
from node.core import build_graph


SEP = "-" * 78
WIDE = "=" * 78


class _NoReporter:
    def attach(self, runtime, root, *, order=None):  # noqa: D401, ANN001
        import contextlib

        return contextlib.nullcontext()


@dataclass
class BenchResult:
    dim_size: int
    depth: int
    workers: int
    total_nodes: int
    build_graph_s: float
    run_s: float
    nodes_per_second: float
    profile_total_s: float
    groups: dict[str, float]


def _header(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def _table(cols: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(c), *(len(r[i]) for r in rows)) for i, c in enumerate(cols)]
    print("  " + "  ".join(f"{c:<{widths[i]}}" for i, c in enumerate(cols)))
    print("  " + "  ".join("-" * w for w in widths))
    for row in rows:
        print("  " + "  ".join(f"{v:<{widths[i]}}" for i, v in enumerate(row)))


def _configure_runtime(workers: int, cache_size: int) -> None:
    node.reset()
    node.configure(
        reporter=_NoReporter(),
        executor="thread",
        workers=workers,
        cache=MemoryLRU(maxsize=cache_size),
        continue_on_error=False,
        validate=False,
    )


def _build_deep_graph(dim_size: int, depth: int):
    @node.dimension(name="bench_dim")
    def bench_dim():
        return list(range(dim_size))

    @node.define(cache=True)
    def lift(x: int) -> int:
        return x

    @node.define(cache=True)
    def add_bias(x: int, bias: int) -> int:
        return x + bias

    @node.define(cache=True)
    def mul_scale(x: int, scale: int) -> int:
        return x * scale

    @node.define(cache=True)
    def fuse(a: int, b: int) -> int:
        return a + b

    @node.define(cache=True, reduce_dims="all")
    def sum_all(arr) -> int:  # noqa: ANN001
        total = 0
        for item in arr.flat:
            total += item
        return total

    layer = lift(x=bench_dim())
    for d in range(depth):
        left = add_bias(x=layer, bias=d + 1)
        right = mul_scale(x=layer, scale=(d % 7) + 2)
        layer = fuse(a=left, b=right)
    root = sum_all(arr=layer)
    return root


def _extract_profile_groups(stats: pstats.Stats) -> tuple[float, dict[str, float]]:
    category_rules: dict[str, list[re.Pattern[str]]] = {
        "completed-task handling": [
            re.compile(r"runtime\.py:.*:release_consumed_deps"),
            re.compile(r"runtime\.py:.*:<dictcomp>"),
            re.compile(r"graphlib\.py:.*:done"),
            re.compile(r"threading\.py:.*:release"),
            re.compile(r"_base\.py:.*:result"),
            re.compile(r"_base\.py:.*:wait"),
        ],
        "ready-node discovery": [
            re.compile(r"graphlib\.py:.*:get_ready"),
        ],
        "ready pre-actions": [
            re.compile(r"runtime\.py:.*:_ensure_deps_ready"),
            re.compile(r"runtime\.py:.*:_resolve"),
            re.compile(r"runtime\.py:.*:_bind_args"),
            re.compile(r"cache\.py:.*:get"),
        ],
        "threadpool submission": [
            re.compile(r"runtime\.py:.*:submit"),
            re.compile(r"thread\.py:.*:submit"),
            re.compile(r"threading\.py:.*:acquire"),
        ],
    }

    grouped: dict[str, float] = {k: 0.0 for k in category_rules}
    total_tt = 0.0

    for func_key, func_stat in stats.stats.items():
        filename, line_no, func_name = func_key
        cc, nc, tt, ct, callers = func_stat
        _ = (cc, nc, ct, callers)
        total_tt += tt
        token = f"{os.path.basename(filename)}:{line_no}:{func_name}"
        for category, patterns in category_rules.items():
            if any(p.search(token) for p in patterns):
                grouped[category] += tt
                break

    return total_tt, grouped


def run_once(dim_size: int, depth: int, workers: int) -> BenchResult:
    cache_size = dim_size * max(8, depth * 4)

    _configure_runtime(workers=workers, cache_size=cache_size)
    root = _build_deep_graph(dim_size=dim_size, depth=depth)

    t0 = time.perf_counter()
    order, _edges = build_graph(root, node.get_runtime().cache)
    build_s = time.perf_counter() - t0
    total_nodes = len(order)

    # 冷运行 profile（首次执行）
    prof = cProfile.Profile()
    prof.enable()
    t0 = time.perf_counter()
    result = root()
    run_s = time.perf_counter() - t0
    prof.disable()
    if result is None:
        raise RuntimeError("benchmark failed: root result is None")

    stats = pstats.Stats(prof).strip_dirs()
    profile_total_s, grouped = _extract_profile_groups(stats)

    return BenchResult(
        dim_size=dim_size,
        depth=depth,
        workers=workers,
        total_nodes=total_nodes,
        build_graph_s=build_s,
        run_s=run_s,
        nodes_per_second=total_nodes / run_s if run_s > 0 else 0.0,
        profile_total_s=profile_total_s,
        groups=grouped,
    )


def run_benchmark(
    dim_size: int,
    depth: int,
    workers: int,
    repeats: int,
) -> None:
    print(f"\n{WIDE}")
    print("  Deep-Dependency Scheduler Benchmark")
    print(f"  dim_size={dim_size:,}, depth={depth}, workers={workers}")
    print(f"  CPU={os.cpu_count()}")
    print(f"{WIDE}")

    runs: list[BenchResult] = []
    for i in range(repeats):
        _header(f"Run {i + 1}/{repeats}")
        run = run_once(
            dim_size=dim_size,
            depth=depth,
            workers=workers,
        )
        runs.append(run)
        print(f"  total nodes: {run.total_nodes:,}")
        print(f"  build_graph: {run.build_graph_s * 1000:.2f} ms")
        print(f"  execute(run): {run.run_s * 1000:.2f} ms")
        print(f"  throughput: {run.nodes_per_second:,.0f} nodes/s")

    med = _median_result(runs)

    _header("Median Result")
    print(f"  total nodes: {med.total_nodes:,}")
    print(f"  build_graph: {med.build_graph_s * 1000:.2f} ms")
    print(f"  execute(run): {med.run_s * 1000:.2f} ms")
    print(f"  throughput: {med.nodes_per_second:,.0f} nodes/s")
    print(f"  profile total self-time: {med.profile_total_s * 1000:.2f} ms")

    rows: list[list[str]] = []
    for name, sec in med.groups.items():
        pct = (sec / med.profile_total_s * 100.0) if med.profile_total_s > 0 else 0.0
        rows.append([name, f"{sec * 1000:.2f}", f"{pct:.1f}%"])
    rows.sort(key=lambda row: float(row[1]), reverse=True)
    _table(["scheduler group", "self-time(ms)", "profile share"], rows)

    _header("How to Read These Numbers")
    print("  1) High 'ready pre-actions' -> optimize cache.get and _resolve/_ensure_deps_ready.")
    print("  2) High 'ready-node discovery' -> get_ready + sorting overhead is amplified.")
    print("  3) High 'threadpool submission' -> submit/acquire overhead approaches task cost.")
    print("  4) High 'completed-task handling' -> future/result/done/release limits throughput.")


def _median_result(runs: list[BenchResult]) -> BenchResult:
    if not runs:
        raise ValueError("empty runs")

    def med(values: list[float]) -> float:
        return statistics.median(values)

    base = runs[0]
    grouped_keys = list(base.groups.keys())
    return BenchResult(
        dim_size=base.dim_size,
        depth=base.depth,
        workers=base.workers,
        total_nodes=int(med([float(r.total_nodes) for r in runs])),
        build_graph_s=med([r.build_graph_s for r in runs]),
        run_s=med([r.run_s for r in runs]),
        nodes_per_second=med([r.nodes_per_second for r in runs]),
        profile_total_s=med([r.profile_total_s for r in runs]),
        groups={k: med([r.groups[k] for r in runs]) for k in grouped_keys},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Scheduler benchmark for deep dependency DAGs")
    parser.add_argument("--dim-size", type=int, default=12_000, help="Dimension size (recommended >= 10000)")
    parser.add_argument("--depth", type=int, default=10, help="Dependency depth")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 8))
    parser.add_argument("--repeats", type=int, default=2, help="Benchmark repeats (median used)")
    args = parser.parse_args()

    run_benchmark(
        dim_size=args.dim_size,
        depth=args.depth,
        workers=args.workers,
        repeats=args.repeats,
    )


if __name__ == "__main__":
    main()
