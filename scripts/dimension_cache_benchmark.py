"""Node 库大规模节点性能基准测试。

用途
----
量化分析 node 库在大规模节点场景下的性能瓶颈。核心问题：
  "即使任务相同，节点越多，每个节点的平均耗时反而更长"——这是 O(N²) 还是其他问题？

测试阶段
--------
  A. scale      规模扩展性：100 → 1k → 10k → 100k 节点，证明超线性拆解
  B. decompose  耗时分解：hash / broadcast / build_graph / topo_sort / exec 各环节
  C. contention 线程同步隔离：serial / thread + 不同缓存的锁竞争
  D. gc         GC 关闭 vs 开启的性能差异
  E. cache      缓存读写（MemoryLRU / DiskCache / ChainCache）
  F. coldwarm   冷热对比
  G. profile    cProfile 热点

运行方式
--------
    uv run python -m scripts.dimension_cache_benchmark
    uv run python -m scripts.dimension_cache_benchmark --section A B C
    uv run python -m scripts.dimension_cache_benchmark --profile
    uv run python -m scripts.dimension_cache_benchmark --keep-cache
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import io
import os
import pstats
import shutil
import statistics
import tempfile
import time
from graphlib import TopologicalSorter
from textwrap import indent
from typing import Any, Callable

import node
from node import ChainCache, DiskCache, MemoryLRU
from node.core import build_graph

# ─────────────────────────────────────────────────────────────────────────────
# 常量 & 辅助
# ─────────────────────────────────────────────────────────────────────────────

_SEP = "─" * 72
_WIDE = "═" * 72
_DEFAULT_CACHE_DIR = ".cache"
_SCALE_SIZES = [100, 1_000, 10_000, 100_000]


def _hdr(title: str) -> None:
    print(f"\n{_SEP}\n  {title}\n{_SEP}")


def _row(label: str, value: str, width: int = 40) -> None:
    print(f"  {label:<{width}} {value}")


def _table_row(cols: list[str], widths: list[int]) -> None:
    parts = [f"{c:<{w}}" for c, w in zip(cols, widths)]
    print("  " + "  ".join(parts))


def _table_header(cols: list[str], widths: list[int]) -> None:
    _table_row(cols, widths)
    sep = "  " + "  ".join("-" * w for w in widths)
    print(sep)


def _bar(ratio: float, width: int = 30) -> str:
    """比例条，ratio 在 [0, 1] 之间。"""
    filled = round(ratio * width)
    return "█" * filled + "░" * (width - filled)


def _trend(values: list[float]) -> str:
    """简单判断趋势：values 是各 size 的每节点耗时（µs）。"""
    if len(values) < 2:
        return "N/A"
    ratios = [values[i + 1] / values[i] if values[i] > 0 else 1 for i in range(len(values) - 1)]
    avg_ratio = statistics.mean(ratios)
    if avg_ratio > 1.5:
        return f"⚠  超线性 ×{avg_ratio:.2f}/step"
    elif avg_ratio > 1.1:
        return f"△  轻微超线性 ×{avg_ratio:.2f}/step"
    else:
        return "✓  线性或更优"


class _NoReporter:
    """禁止 RichReporter 的进度条输出。"""

    def attach(self, runtime: Any, root: Any):
        import contextlib
        return contextlib.nullcontext()


def _reset(**kw: Any) -> None:
    """reset runtime 并用给定参数重新配置（默认禁用 reporter & DiskCache）。"""
    node.reset()
    kw.setdefault("reporter", _NoReporter())
    kw.setdefault("cache", MemoryLRU(65536))
    node.configure(**kw)


def _timed(fn: Callable, repeats: int = 3) -> float:
    """运行 fn 若干次，取中位数耗时（秒）。"""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def _cleanup_default_cache() -> None:
    """删除默认磁盘缓存目录，确保测试结果不受历史缓存影响。"""
    cache_dir = _DEFAULT_CACHE_DIR
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)
        print(f"  [清理] 已删除磁盘缓存目录: {os.path.abspath(cache_dir)}")
    else:
        print(f"  [清理] 缓存目录不存在，无需清理: {os.path.abspath(cache_dir)}")


# ─────────────────────────────────────────────────────────────────────────────
# 阶段 A：规模扩展性分析
# ─────────────────────────────────────────────────────────────────────────────

def bench_scale(sizes: list[int] | None = None) -> None:
    """多规模节点下的扩展性，定量证明超线性瓶颈。"""
    _hdr("阶段 A：规模扩展性分析（noop identity, serial, 纯内存缓存）")
    sizes = sizes or _SCALE_SIZES

    cols = ["Size", "Build(ms)", "build_graph(ms)", "Exec(ms)",
            "Total(ms)", "Per-Node(µs)", "偏离×基准"]
    widths = [9, 11, 17, 10, 11, 14, 12]
    _table_header(cols, widths)

    baseline_per_node: float | None = None
    per_node_list: list[float] = []

    for size in sizes:
        _reset(executor="thread", workers=1)

        @node.dimension(name="dim_scale")
        def dim():
            return list(range(size))

        @node.define(cache=False)
        def noop(x: int) -> int:
            return x

        # 1) 构建耗时
        t0 = time.perf_counter()
        dim_node = dim()
        vec = noop(x=dim_node)
        dt_build = time.perf_counter() - t0

        # 2) build_graph 耗时
        t0 = time.perf_counter()
        order, edges = build_graph(vec, cache=None)
        dt_graph = time.perf_counter() - t0

        # 3) 执行耗时（取 median 3 轮，但 size 大时仅 1 轮避免超时）
        repeats = 1 if size >= 50_000 else 3

        def _exec():
            node.reset()
            node.configure(reporter=_NoReporter(), cache=MemoryLRU(size + 10),
                           executor="thread", workers=1)

            @node.dimension(name="dim_scale_e")
            def dim2():
                return list(range(size))

            @node.define(cache=False)
            def noop2(x: int) -> int:
                return x

            noop2(x=dim2())()

        dt_exec = _timed(_exec, repeats)
        dt_total = dt_build + dt_graph + dt_exec
        per_node = dt_total / size * 1e6  # µs

        if baseline_per_node is None:
            baseline_per_node = per_node
        per_node_list.append(per_node)
        deviation = per_node / baseline_per_node if baseline_per_node else 1.0

        _table_row([
            f"{size:,}",
            f"{dt_build * 1000:.2f}",
            f"{dt_graph * 1000:.2f}",
            f"{dt_exec * 1000:.2f}",
            f"{dt_total * 1000:.2f}",
            f"{per_node:.2f}",
            f"{deviation:.2f}×",
        ], widths)

    print(f"\n  趋势：{_trend(per_node_list)}")
    print(f"  说明：偏离×基准 > 1 即证明存在超线性开销（每节点越来越慢）")


# ─────────────────────────────────────────────────────────────────────────────
# 阶段 B：逐环节耗时分解
# ─────────────────────────────────────────────────────────────────────────────

def bench_decompose(size: int = 1000) -> None:
    """分别计时各关键环节，量化每个阶段在总耗时中的占比。"""
    _hdr(f"阶段 B：逐环节耗时分解（size={size:,}）")

    _reset(executor="thread", workers=1)

    @node.dimension(name="dim_decomp")
    def dim():
        return list(range(size))

    @node.define(cache=True)
    def identity(x: int) -> int:
        return x

    # B1：维度节点构建
    t0 = time.perf_counter()
    dim_node = dim()
    dt_dim = time.perf_counter() - t0

    # B2：广播展开（broadcast）
    t0 = time.perf_counter()
    vec = identity(x=dim_node)
    dt_broadcast = time.perf_counter() - t0

    # B3：build_graph (无缓存)
    t0 = time.perf_counter()
    order, edges = build_graph(vec, cache=None)
    dt_build_graph = time.perf_counter() - t0

    # B4：拓扑排序（单独计时 TopologicalSorter）
    t0 = time.perf_counter()
    _ = list(TopologicalSorter(edges).static_order())
    dt_topo = time.perf_counter() - t0

    # B5：执行（冷缓存）— 重置确保无缓存
    _reset(executor="thread", workers=1, cache=MemoryLRU(size + 10))

    @node.dimension(name="dim_decomp2")
    def dim2():
        return list(range(size))

    @node.define(cache=True)
    def identity2(x: int) -> int:
        return x

    vec2 = identity2(x=dim2())
    t0 = time.perf_counter()
    vec2()
    dt_exec_cold = time.perf_counter() - t0

    # B6：执行（热缓存）
    t0 = time.perf_counter()
    vec2()
    dt_exec_warm = time.perf_counter() - t0

    phases = [
        ("dimension() 构建", dt_dim),
        ("broadcast 展开 (N Node)", dt_broadcast),
        ("build_graph (DFS + edges)", dt_build_graph),
        ("TopologicalSorter.static_order", dt_topo),
        ("执行（冷缓存）", dt_exec_cold),
        ("执行（热缓存）", dt_exec_warm),
    ]

    # 取非热缓存执行的总和作为"一次完整 cold run"
    total_cold = dt_dim + dt_broadcast + dt_exec_cold
    total = sum(dt for _, dt in phases)

    print()
    w_label, w_ms, w_us, w_bar = 35, 11, 14, 30
    _table_header(
        ["环节", "耗时(ms)", "Per-Node(µs)", "占比"],
        [w_label, w_ms, w_us, w_bar + 6]
    )
    for label, dt in phases:
        ratio = dt / total if total > 0 else 0
        _table_row([
            label,
            f"{dt * 1000:.3f}",
            f"{dt / size * 1e6:.2f}",
            f"{_bar(ratio, w_bar)}  {ratio * 100:.1f}%",
        ], [w_label, w_ms, w_us, w_bar + 6])

    print(f"\n  冷运行总计：{total_cold * 1000:.2f} ms  ({total_cold / size * 1e6:.2f} µs/节点)")
    print(f"  build_graph 占冷运行比：{dt_build_graph / total_cold * 100:.1f}%")
    print(f"  broadcast 占冷运行比：  {dt_broadcast / total_cold * 100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 阶段 C：线程同步与锁竞争隔离
# ─────────────────────────────────────────────────────────────────────────────

def bench_contention(size: int = 2000) -> None:
    """对比不同执行器 + 缓存组合下的锁竞争开销。"""
    _hdr(f"阶段 C：线程同步与锁竞争隔离（size={size:,}）")

    cpu = os.cpu_count() or 4
    configs: list[tuple[str, dict[str, Any]]] = [
        ("serial  w=1  无缓存", dict(executor="thread", workers=1,
                                    cache=MemoryLRU(1))),
        ("serial  w=1  MemoryLRU", dict(executor="thread", workers=1,
                                        cache=MemoryLRU(size + 100))),
        (f"thread  w=2  MemoryLRU", dict(executor="thread", workers=2,
                                         cache=MemoryLRU(size + 100))),
        (f"thread  w={min(4, cpu)}  MemoryLRU", dict(executor="thread", workers=min(4, cpu),
                                                     cache=MemoryLRU(size + 100))),
        (f"thread  w={min(cpu, 8)}  MemoryLRU", dict(executor="thread", workers=min(cpu, 8),
                                                      cache=MemoryLRU(size + 100))),
    ]

    widths = [30, 12, 14, 14]
    _table_header(["配置", "Total(ms)", "Per-Node(µs)", "vs serial×1"], widths)

    baseline: float | None = None
    for label, cfg in configs:
        node.reset()
        node.configure(reporter=_NoReporter(), **cfg)

        @node.dimension(name="dim_cont")
        def dim():
            return list(range(size))

        @node.define(cache=False)
        def noop_c(x: int) -> int:
            return x

        vec = noop_c(x=dim())
        dt = _timed(lambda: vec(), repeats=3)  # noqa: B023

        if baseline is None:
            baseline = dt
        ratio = dt / baseline if baseline else 1.0

        _table_row([
            label,
            f"{dt * 1000:.2f}",
            f"{dt / size * 1e6:.2f}",
            f"{ratio:.2f}×",
        ], widths)

    print(f"\n  说明：理想并行应 <1×；若 >1× 说明线程调度开销 > 并行收益")


# ─────────────────────────────────────────────────────────────────────────────
# 阶段 D：GC 压力分析
# ─────────────────────────────────────────────────────────────────────────────

def bench_gc(size: int = 5000) -> None:
    """GC 关闭 vs 开启的性能差异——衡量大量 Node 对象的 GC 压力。"""
    _hdr(f"阶段 D：GC 压力分析（size={size:,}）")

    gc_stats_before = gc.get_count()

    def _run(with_gc: bool) -> float:
        if not with_gc:
            gc.disable()
        else:
            gc.enable()

        try:
            _reset(executor="thread", workers=1, cache=MemoryLRU(size + 10))

            @node.dimension(name="dim_gc")
            def dim():
                return list(range(size))

            @node.define(cache=False)
            def noop_g(x: int) -> int:
                return x

            vec = noop_g(x=dim())
            gc.collect()  # 清除构建阶段产生的垃圾

            t0 = time.perf_counter()
            vec()
            return time.perf_counter() - t0
        finally:
            gc.enable()

    dt_gc_on = _run(with_gc=True)
    dt_gc_off = _run(with_gc=False)

    # GC 触发统计
    gc_stats_after = gc.get_count()
    gc_diff = tuple(a - b for a, b in zip(gc_stats_after, gc_stats_before))

    _row("GC 开启耗时", f"{dt_gc_on * 1000:.2f} ms  ({dt_gc_on / size * 1e6:.2f} µs/节点)")
    _row("GC 关闭耗时", f"{dt_gc_off * 1000:.2f} ms  ({dt_gc_off / size * 1e6:.2f} µs/节点)")
    speedup = dt_gc_on / max(dt_gc_off, 1e-9)
    _row("GC 开销倍数", f"{speedup:.2f}×  (>1.10 说明 GC 有显著影响)")
    _row("GC 统计增量 (gen0/1/2)", str(gc_diff))


# ─────────────────────────────────────────────────────────────────────────────
# 阶段 E：缓存读写基准（原阶段 3）
# ─────────────────────────────────────────────────────────────────────────────

def bench_cache_rw(size: int) -> None:
    """MemoryLRU / DiskCache / ChainCache 读写性能对比。"""
    _hdr("阶段 E：缓存读写基准")

    n = min(size, 1000)
    keys = list(range(n))
    value = list(range(100))

    tmp = tempfile.mkdtemp(prefix="node_bench_cache_")
    try:
        caches: list[tuple[str, Any]] = [
            ("MemoryLRU", MemoryLRU(maxsize=n + 10)),
            ("DiskCache", DiskCache(os.path.join(tmp, "disk"))),
            ("ChainCache(Mem+Disk)", ChainCache([
                MemoryLRU(maxsize=n + 10),
                DiskCache(os.path.join(tmp, "chain")),
            ])),
        ]
        widths = [28, 14, 14, 14]
        _table_header(["缓存类型", "put(µs/op)", "get(命中)(µs/op)", "get(未命中)(µs/op)"], widths)
        for name, cache in caches:
            t0 = time.perf_counter()
            for k in keys:
                cache.put("fn", k, value)
            dt_put = (time.perf_counter() - t0) / n * 1e6

            t0 = time.perf_counter()
            for k in keys:
                cache.get("fn", k)
            dt_hit = (time.perf_counter() - t0) / n * 1e6

            t0 = time.perf_counter()
            for k in range(n, 2 * n):
                cache.get("fn", k)
            dt_miss = (time.perf_counter() - t0) / n * 1e6

            _table_row([name, f"{dt_put:.2f}", f"{dt_hit:.2f}", f"{dt_miss:.2f}"], widths)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# 阶段 F：冷热缓存对比
# ─────────────────────────────────────────────────────────────────────────────

def bench_cold_warm(size: int) -> None:
    """冷缓存（首次计算）vs 热缓存（二次命中）时间比。"""
    _hdr("阶段 F：冷热缓存对比")

    tmp = tempfile.mkdtemp(prefix="node_bench_coldwarm_")
    try:
        cache_configs: list[tuple[str, Any]] = [
            ("MemoryLRU only", MemoryLRU(maxsize=size + 100)),
            ("ChainCache(Mem+Disk)", ChainCache([
                MemoryLRU(maxsize=size + 100),
                DiskCache(os.path.join(tmp, "chain")),
            ])),
        ]
        widths = [32, 12, 12, 10]
        _table_header(["缓存", "冷运行(ms)", "热运行(ms)", "加速比"], widths)
        for cache_label, cache_obj in cache_configs:
            dt_cold = 0.0
            for pass_name in ("冷", "热"):
                _reset(cache=cache_obj, executor="thread", workers=4)

                @node.dimension(name="dim_cw")
                def dim():
                    return list(range(size))

                @node.define(cache=True)
                def identity_cw(x: int) -> int:
                    return x

                vec = identity_cw(x=dim())
                t0 = time.perf_counter()
                vec()
                dt = time.perf_counter() - t0

                if pass_name == "冷":
                    dt_cold = dt
                else:
                    speedup = dt_cold / max(dt, 1e-9)
                    _table_row([
                        cache_label,
                        f"{dt_cold * 1000:.2f}",
                        f"{dt * 1000:.2f}",
                        f"{speedup:.1f}×",
                    ], widths)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# 阶段 G：cProfile 热点分析
# ─────────────────────────────────────────────────────────────────────────────

def bench_profile(size: int, topn: int = 30) -> None:
    """对冷运行进行 cProfile 分析，找出 CPU 热点。"""
    _hdr(f"阶段 G：cProfile 热点分析（冷运行, size={size:,}, top {topn}）")

    _reset(executor="thread", workers=4, cache=MemoryLRU(size + 10))

    @node.dimension(name="dim_prof")
    def dim():
        return list(range(size))

    @node.define(cache=True)
    def identity_p(x: int) -> int:
        return x

    vec = identity_p(x=dim())

    prof = cProfile.Profile()
    prof.enable()
    t0 = time.perf_counter()
    vec()
    dt = time.perf_counter() - t0
    prof.disable()
    _row("冷运行耗时", f"{dt * 1000:.2f} ms  ({dt / size * 1e6:.2f} µs/节点)")

    buf = io.StringIO()
    ps = pstats.Stats(prof, stream=buf).strip_dirs().sort_stats("tottime")
    ps.print_stats(topn)
    print(indent(buf.getvalue(), "  "))


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

SECTIONS: dict[str, Callable] = {
    "A": bench_scale,
    "B": bench_decompose,
    "C": bench_contention,
    "D": bench_gc,
    "E": bench_cache_rw,
    "F": bench_cold_warm,
    "G": bench_profile,
}


def run_benchmark(
    size: int = 1000,
    sections: list[str] | None = None,
    profile: bool = False,
    keep_cache: bool = False,
) -> None:
    """运行选定的基准测试阶段。"""
    enabled = set(sections) if sections else set(SECTIONS) - {"G"}
    if profile:
        enabled.add("G")

    print(f"\n{_WIDE}")
    print(f"  Node 大规模性能基准测试   size={size:,}   PID={os.getpid()}")
    print(f"  CPU 核数: {os.cpu_count()}")
    print(f"{_WIDE}")

    if not keep_cache:
        print()
        _cleanup_default_cache()

    # 按字母顺序执行各阶段
    for key in sorted(enabled):
        fn = SECTIONS[key]
        # 阶段 A：不传 size，使用内置规模列表
        if key == "A":
            fn()
        else:
            fn(size)  # type: ignore[call-arg]

    print(f"\n{_WIDE}")
    print("  基准测试完成")
    print(f"{_WIDE}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Node 大规模节点性能基准测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
测试阶段:
  A  规模扩展性（100 → 1k → 10k → 100k）
  B  耗时分解（hash / broadcast / build_graph / exec）
  C  线程同步隔离（锁竞争）
  D  GC 压力分析
  E  缓存读写（MemoryLRU/DiskCache/ChainCache）
  F  冷热缓存对比
  G  cProfile 热点分析

示例:
  uv run python -m scripts.dimension_cache_benchmark
  uv run python -m scripts.dimension_cache_benchmark --section A B C
  uv run python -m scripts.dimension_cache_benchmark --profile
  uv run python -m scripts.dimension_cache_benchmark --keep-cache
""",
    )
    parser.add_argument("--size", type=int, default=1000, help="基准维度大小（默认 1000）")
    parser.add_argument(
        "--section",
        nargs="+",
        choices=list(SECTIONS),
        metavar="SECTION",
        help="仅运行指定阶段（默认全部，除 G）",
    )
    parser.add_argument("--profile", action="store_true", help="额外运行 cProfile 热点分析")
    parser.add_argument("--keep-cache", action="store_true", help="保留已有磁盘缓存（默认删除）")
    args = parser.parse_args()
    run_benchmark(
        size=args.size,
        sections=args.section,
        profile=args.profile,
        keep_cache=args.keep_cache,
    )


if __name__ == "__main__":
    main()
