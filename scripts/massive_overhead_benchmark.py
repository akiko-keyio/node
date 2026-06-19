"""超大规模框架开销基准：复杂依赖 × 多维广播 × 深度嵌套。

每个节点函数均为 noop（直接返回），隔离纯框架开销。
分析维度：调度路径、logger/reporter、缓存、哈希、广播展开。

运行示例
--------
uv run python -m scripts.massive_overhead_benchmark
uv run python -m scripts.massive_overhead_benchmark --dim-a 60 --dim-b 50 --dim-c 30 --depth 8
uv run python -m scripts.massive_overhead_benchmark --section A B C D E F
uv run python -m scripts.massive_overhead_benchmark --profile
"""

from __future__ import annotations

import argparse
import contextlib
import cProfile
import gc
import io
import os
import pstats
import re
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from textwrap import indent
from typing import Any, Callable

# Windows GBK workaround
if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

# ─────────────────────────────────────────────────────────────────────────────
# 输出辅助
# ─────────────────────────────────────────────────────────────────────────────

_SEP = "─" * 78
_WIDE = "═" * 78


def _hdr(title: str) -> None:
    print(f"\n{_SEP}\n  {title}\n{_SEP}")


def _row(label: str, value: str, w: int = 44) -> None:
    print(f"  {label:<{w}} {value}")


def _tbl_hdr(cols: list[str], ws: list[int]) -> None:
    print("  " + "  ".join(f"{c:<{w}}" for c, w in zip(cols, ws)))
    print("  " + "  ".join("-" * w for w in ws))


def _tbl_row(vals: list[str], ws: list[int]) -> None:
    print("  " + "  ".join(f"{v:<{w}}" for v, w in zip(vals, ws)))


def _bar(ratio: float, width: int = 28) -> str:
    filled = round(ratio * width)
    return "█" * filled + "░" * (width - filled)


class _NoReporter:
    def attach(self, runtime, root, *, order=None):  # noqa: ANN001
        return contextlib.nullcontext()


# ─────────────────────────────────────────────────────────────────────────────
# 运行时管理
# ─────────────────────────────────────────────────────────────────────────────

def _reset(**kw: Any) -> None:
    import node
    from node import MemoryLRU

    node.reset()
    kw.setdefault("reporter", _NoReporter())
    kw.setdefault("validate", False)
    kw.setdefault("cache", MemoryLRU(500_000))
    node.configure(**kw)


def _timed(fn: Callable, repeats: int = 1) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


# ─────────────────────────────────────────────────────────────────────────────
# 图构建：超大规模复杂依赖拓扑
#
# 拓扑结构：
#   Layer 0: 3 个维度节点 dim_a(A) × dim_b(B) × dim_c(C)
#   Layer 1: lift(x=dim_abc) → A×B×C 个标量节点
#   Layer 2..D: 每层菱形模式:
#       left  = transform_l(x=prev, bias=layer_id)
#       right = transform_r(x=prev, scale=layer_id)
#       fused = fuse(a=left, b=right)
#   Layer D+1: reduce_dims("dim_c") → A×B 个节点
#   Layer D+2: reduce_dims("dim_b") → A 个节点
#   Layer D+3: reduce_dims("all")   → 1 个标量节点
#
# 总节点数 ≈ A×B×C × (1 + 3×D) + A×B + A + 1 + 维度节点
# ─────────────────────────────────────────────────────────────────────────────

def _build_massive_graph(
    dim_a: int, dim_b: int, dim_c: int, depth: int
) -> tuple[Any, int]:
    """构建超大规模 DAG 并返回 (root_node, estimated_node_count)。"""
    import node

    @node.dimension(name="dim_a")
    def make_dim_a():
        return list(range(dim_a))

    @node.dimension(name="dim_b")
    def make_dim_b():
        return list(range(dim_b))

    @node.dimension(name="dim_c")
    def make_dim_c():
        return list(range(dim_c))

    @node.define(cache=False)
    def lift(a: int, b: int, c: int) -> int:
        return 0

    @node.define(cache=False)
    def transform_l(x: int, bias: int) -> int:
        return x

    @node.define(cache=False)
    def transform_r(x: int, scale: int) -> int:
        return x

    @node.define(cache=False)
    def fuse(a: int, b: int) -> int:
        return a

    @node.define(cache=False, reduce_dims=["dim_c"])
    def reduce_c(arr) -> int:  # noqa: ANN001
        return 0

    @node.define(cache=False, reduce_dims=["dim_b"])
    def reduce_b(arr) -> int:  # noqa: ANN001
        return 0

    @node.define(cache=False, reduce_dims="all")
    def reduce_all(arr) -> int:  # noqa: ANN001
        return 0

    da = make_dim_a()
    db = make_dim_b()
    dc = make_dim_c()

    layer = lift(a=da, b=db, c=dc)

    for d in range(depth):
        left = transform_l(x=layer, bias=d + 1)
        right = transform_r(x=layer, scale=(d % 5) + 2)
        layer = fuse(a=left, b=right)

    stage_rc = reduce_c(arr=layer)
    stage_rb = reduce_b(arr=stage_rc)
    root = reduce_all(arr=stage_rb)

    abc = dim_a * dim_b * dim_c
    ab = dim_a * dim_b
    estimated = abc * (1 + 3 * depth) + ab + dim_a + 1 + (dim_a + dim_b + dim_c)
    return root, estimated


# ─────────────────────────────────────────────────────────────────────────────
# 阶段 A：节点构建 + 哈希 + 广播开销
# ─────────────────────────────────────────────────────────────────────────────

def bench_build(dim_a: int, dim_b: int, dim_c: int, depth: int) -> None:
    _hdr(f"阶段 A：节点构建/哈希/广播开销（{dim_a}×{dim_b}×{dim_c}, depth={depth}）")
    _reset(executor="thread", workers=1)

    t0 = time.perf_counter()
    root, est = _build_massive_graph(dim_a, dim_b, dim_c, depth)
    dt_build = time.perf_counter() - t0

    _row("预估节点数", f"{est:,}")
    _row("节点构建耗时", f"{dt_build * 1000:.1f} ms")
    _row("每节点构建(含hash+broadcast)", f"{dt_build / est * 1e6:.2f} µs")

    from node.core import build_graph as bg

    t0 = time.perf_counter()
    order, edges = bg(root, cache=None)
    dt_graph = time.perf_counter() - t0
    actual = len(order)

    _row("build_graph 耗时", f"{dt_graph * 1000:.1f} ms")
    _row("实际节点数", f"{actual:,}")
    _row("每节点 build_graph", f"{dt_graph / actual * 1e6:.2f} µs")


# ─────────────────────────────────────────────────────────────────────────────
# 阶段 B：串行 vs 并行执行开销（noop 节点）
# ─────────────────────────────────────────────────────────────────────────────

def bench_exec_modes(dim_a: int, dim_b: int, dim_c: int, depth: int) -> None:
    _hdr(f"阶段 B：执行模式对比（noop, {dim_a}×{dim_b}×{dim_c}, depth={depth}）")

    cpu = os.cpu_count() or 4
    configs = [
        ("serial  w=1", 1),
        ("thread  w=2", 2),
        (f"thread  w={min(4, cpu)}", min(4, cpu)),
        (f"thread  w={min(cpu, 8)}", min(cpu, 8)),
    ]

    ws = [24, 12, 14, 14]
    _tbl_hdr(["执行模式", "Total(ms)", "Per-Node(µs)", "vs serial"], ws)

    baseline: float | None = None
    for label, workers in configs:
        _reset(executor="thread", workers=workers, cache=None)
        root, est = _build_massive_graph(dim_a, dim_b, dim_c, depth)

        t0 = time.perf_counter()
        root()
        dt = time.perf_counter() - t0

        from node.core import build_graph as bg
        order, _ = bg(root, cache=None)
        actual = len(order)

        if baseline is None:
            baseline = dt
        ratio = dt / baseline if baseline else 1.0

        _tbl_row([label, f"{dt * 1000:.1f}", f"{dt / actual * 1e6:.2f}", f"{ratio:.2f}×"], ws)

    print(f"\n  说明：noop 节点下 ratio >1 说明线程调度开销 > 并行收益")


# ─────────────────────────────────────────────────────────────────────────────
# 阶段 C：Logger 开销隔离
# ─────────────────────────────────────────────────────────────────────────────

def _disable_loguru() -> Any:
    """临时禁用所有 loguru handler，返回恢复所需信息。"""
    from loguru import logger as _lg

    handler_ids = list(_lg._core.handlers.keys())
    for hid in handler_ids:
        try:
            _lg.remove(hid)
        except ValueError:
            pass
    return handler_ids


def _restore_loguru(handler_ids: Any) -> None:
    """恢复 loguru 原始 handler。"""
    from node.logger import logger as _lg, console as _con
    from rich.logging import RichHandler

    _lg.remove()
    _lg.add(
        RichHandler(
            console=_con, markup=True,
            show_time=True, show_level=True, show_path=True,
        ),
        level="DEBUG",
        format="{message}",
    )


def bench_logger(dim_a: int, dim_b: int, dim_c: int, depth: int) -> None:
    _hdr(f"阶段 C：Logger/Reporter 开销隔离（{dim_a}×{dim_b}×{dim_c}, depth={depth}）")

    results: list[tuple[str, float, int]] = []

    # C1: 无 reporter, logger 正常 (DEBUG + RichHandler)
    _reset(executor="thread", workers=1, cache=None)
    root, _ = _build_massive_graph(dim_a, dim_b, dim_c, depth)
    from node.core import build_graph as bg
    order, _ = bg(root, cache=None)
    actual = len(order)

    gc.collect()
    t0 = time.perf_counter()
    root()
    dt1 = time.perf_counter() - t0
    results.append(("Logger=DEBUG+Rich, Reporter=No", dt1, actual))

    # C2: 无 reporter, logger 禁用
    _reset(executor="thread", workers=1, cache=None)
    root, _ = _build_massive_graph(dim_a, dim_b, dim_c, depth)
    _disable_loguru()

    gc.collect()
    t0 = time.perf_counter()
    root()
    dt2 = time.perf_counter() - t0
    results.append(("Logger=Disabled, Reporter=No", dt2, actual))

    _restore_loguru(None)

    # C3: 有 RichReporter, logger 正常
    import node as _n
    _reset(executor="thread", workers=1, cache=None)
    root, _ = _build_massive_graph(dim_a, dim_b, dim_c, depth)

    try:
        from node.reporter import RichReporter
        rp = RichReporter(refresh_per_second=1)
        rt = _n.get_runtime()
        rt.reporter = rp

        gc.collect()
        t0 = time.perf_counter()
        root()
        dt3 = time.perf_counter() - t0
        results.append(("Logger=DEBUG+Rich, Reporter=Rich", dt3, actual))
    except Exception as e:
        results.append(("Logger=DEBUG+Rich, Reporter=Rich", -1, actual))
        print(f"  ⚠ RichReporter 测试跳过: {e}")

    # C4: callback 开销 — 模拟 on_node_start/on_node_end
    _reset(executor="thread", workers=1, cache=None)
    root, _ = _build_massive_graph(dim_a, dim_b, dim_c, depth)
    rt = _n.get_runtime()
    _counter = [0]

    def _on_start(n):  # noqa: ANN001
        _counter[0] += 1

    def _on_end(n, dur, cached, failed):  # noqa: ANN001
        _counter[0] += 1

    def _on_state(n, state):  # noqa: ANN001
        _counter[0] += 1

    rt.on_node_start = _on_start
    rt.on_node_end = _on_end
    rt.on_node_state = _on_state

    gc.collect()
    t0 = time.perf_counter()
    root()
    dt4 = time.perf_counter() - t0
    results.append((f"Logger=off, Callbacks=noop({_counter[0]:,})", dt4, actual))

    _restore_loguru(None)

    # 输出
    ws = [42, 12, 14, 10]
    _tbl_hdr(["配置", "Total(ms)", "Per-Node(µs)", "vs baseline"], ws)
    baseline = results[1][1]  # disabled logger as baseline
    for label, dt, cnt in results:
        if dt < 0:
            _tbl_row([label, "SKIP", "-", "-"], ws)
            continue
        ratio = dt / baseline if baseline > 0 else 1.0
        _tbl_row([label, f"{dt * 1000:.1f}", f"{dt / cnt * 1e6:.2f}", f"{ratio:.2f}×"], ws)

    print(f"\n  说明：ratio >1.05 表示该配置引入了可测量的额外开销")


# ─────────────────────────────────────────────────────────────────────────────
# 阶段 D：逐环节耗时分解（含 _set_node_state / _resolve / _bind_args）
# ─────────────────────────────────────────────────────────────────────────────

def bench_decompose(dim_a: int, dim_b: int, dim_c: int, depth: int) -> None:
    _hdr(f"阶段 D：逐环节 cProfile 分解（{dim_a}×{dim_b}×{dim_c}, depth={depth}）")

    _reset(executor="thread", workers=1, cache=None)
    root, _ = _build_massive_graph(dim_a, dim_b, dim_c, depth)

    prof = cProfile.Profile()
    prof.enable()
    t0 = time.perf_counter()
    root()
    dt = time.perf_counter() - t0
    prof.disable()

    from node.core import build_graph as bg
    order, _ = bg(root, cache=None)
    actual = len(order)

    _row("冷执行耗时", f"{dt * 1000:.1f} ms  ({dt / actual * 1e6:.2f} µs/节点)")

    stats = pstats.Stats(prof).strip_dirs()
    total_tt, groups = _extract_groups(stats)

    _row("profile self-time 总计", f"{total_tt * 1000:.1f} ms")
    print()

    ws = [36, 14, 10, 36]
    _tbl_hdr(["开销类别", "self-time(ms)", "占比", "分布"], ws)

    rows = [(name, sec) for name, sec in groups.items()]
    rows.sort(key=lambda r: r[1], reverse=True)
    for name, sec in rows:
        pct = sec / total_tt * 100 if total_tt > 0 else 0
        _tbl_row([name, f"{sec * 1000:.2f}", f"{pct:.1f}%", _bar(pct / 100)], ws)


def _extract_groups(stats: pstats.Stats) -> tuple[float, dict[str, float]]:
    rules: dict[str, list[re.Pattern[str]]] = {
        "scheduling (topo/done/ready)": [
            re.compile(r"graphlib\.py:.*:done"),
            re.compile(r"graphlib\.py:.*:get_ready"),
            re.compile(r"graphlib\.py:.*:prepare"),
            re.compile(r"graphlib\.py:.*:static_order"),
        ],
        "threadpool (submit/wait/result)": [
            re.compile(r"_base\.py:.*:submit"),
            re.compile(r"_base\.py:.*:result"),
            re.compile(r"_base\.py:.*:wait"),
            re.compile(r"thread\.py:.*:submit"),
            re.compile(r"threading\.py:.*:acquire"),
            re.compile(r"threading\.py:.*:release"),
        ],
        "eval_node 框架路径": [
            re.compile(r"runtime\.py:.*:_eval_node"),
        ],
        "_resolve (依赖解析)": [
            re.compile(r"runtime\.py:.*:_resolve"),
        ],
        "_bind_args (参数绑定)": [
            re.compile(r"runtime\.py:.*:_bind_args"),
        ],
        "_ensure_deps_ready": [
            re.compile(r"runtime\.py:.*:_ensure_deps_ready"),
        ],
        "_set_node_state (回调)": [
            re.compile(r"runtime\.py:.*:_set_node_state"),
        ],
        "_save_result + cache.put": [
            re.compile(r"runtime\.py:.*:_save_result"),
            re.compile(r"cache\.py:.*:put"),
        ],
        "cache.get": [
            re.compile(r"cache\.py:.*:get"),
            re.compile(r"runtime\.py:.*:_cache_enabled_for"),
        ],
        "release_deps (内存回收)": [
            re.compile(r"runtime\.py:.*:release_deps"),
            re.compile(r"runtime\.py:.*:release_consumed_deps"),
            re.compile(r"runtime\.py:.*:release_deps_ts"),
        ],
        "hashing (blake2b/_canonical)": [
            re.compile(r"hashing\.py:.*:compute_node_hash"),
            re.compile(r"hashing\.py:.*:_canonical"),
            re.compile(r"hashlib"),
        ],
        "build_graph (DFS)": [
            re.compile(r"core\.py:.*:build_graph"),
        ],
        "Node.__init__ + broadcast": [
            re.compile(r"core\.py:.*:__init__"),
            re.compile(r"dimension\.py:.*:broadcast"),
            re.compile(r"dimension\.py:.*:_resolve_broadcast"),
            re.compile(r"core\.py:.*:_collect_nodes"),
        ],
        "logger/loguru/rich": [
            re.compile(r"logger\.py"),
            re.compile(r"_logger\.py"),
            re.compile(r"rich"),
            re.compile(r"loguru"),
        ],
        "user fn (noop)": [
            re.compile(r"massive_overhead_benchmark\.py:.*:lift"),
            re.compile(r"massive_overhead_benchmark\.py:.*:transform_"),
            re.compile(r"massive_overhead_benchmark\.py:.*:fuse"),
            re.compile(r"massive_overhead_benchmark\.py:.*:reduce_"),
        ],
    }

    grouped: dict[str, float] = {k: 0.0 for k in rules}
    total_tt = 0.0

    for func_key, func_stat in stats.stats.items():
        filename, line_no, func_name = func_key
        cc, nc, tt, ct, callers = func_stat
        total_tt += tt
        token = f"{os.path.basename(filename)}:{line_no}:{func_name}"
        for cat, patterns in rules.items():
            if any(p.search(token) for p in patterns):
                grouped[cat] += tt
                break

    return total_tt, grouped


# ─────────────────────────────────────────────────────────────────────────────
# 阶段 E：cProfile Top-N 热点
# ─────────────────────────────────────────────────────────────────────────────

def bench_profile(dim_a: int, dim_b: int, dim_c: int, depth: int, topn: int = 40) -> None:
    _hdr(f"阶段 E：cProfile 热点 Top-{topn}（{dim_a}×{dim_b}×{dim_c}, depth={depth}）")

    _reset(executor="thread", workers=1, cache=None)
    root, _ = _build_massive_graph(dim_a, dim_b, dim_c, depth)

    prof = cProfile.Profile()
    prof.enable()
    t0 = time.perf_counter()
    root()
    dt = time.perf_counter() - t0
    prof.disable()

    from node.core import build_graph as bg
    order, _ = bg(root, cache=None)
    _row("冷执行耗时", f"{dt * 1000:.1f} ms  ({dt / len(order) * 1e6:.2f} µs/节点, {len(order):,} nodes)")

    buf = io.StringIO()
    ps = pstats.Stats(prof, stream=buf).strip_dirs().sort_stats("tottime")
    ps.print_stats(topn)
    print(indent(buf.getvalue(), "  "))


# ─────────────────────────────────────────────────────────────────────────────
# 阶段 F：规模扩展趋势
# ─────────────────────────────────────────────────────────────────────────────

def bench_scaling(depth: int) -> None:
    _hdr(f"阶段 F：规模扩展趋势（depth={depth}, serial w=1）")

    scales = [
        (10, 10, 10),
        (20, 20, 15),
        (30, 30, 20),
        (40, 40, 25),
        (50, 50, 30),
    ]

    ws = [18, 12, 12, 12, 14, 10]
    _tbl_hdr(["dims (A×B×C)", "est. nodes", "Build(ms)", "Exec(ms)", "Per-Node(µs)", "趋势×"], ws)

    baseline_pn: float | None = None
    for a, b, c in scales:
        _reset(executor="thread", workers=1, cache=None)

        t0 = time.perf_counter()
        root, est = _build_massive_graph(a, b, c, depth)
        dt_build = time.perf_counter() - t0

        gc.collect()
        t0 = time.perf_counter()
        root()
        dt_exec = time.perf_counter() - t0

        per_node = (dt_build + dt_exec) / est * 1e6
        if baseline_pn is None:
            baseline_pn = per_node
        ratio = per_node / baseline_pn

        _tbl_row([
            f"{a}×{b}×{c}",
            f"{est:,}",
            f"{dt_build * 1000:.1f}",
            f"{dt_exec * 1000:.1f}",
            f"{per_node:.2f}",
            f"{ratio:.2f}×",
        ], ws)

    print(f"\n  说明：趋势× >1.2 表示存在超线性开销 (每节点越来越慢)")


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

SECTIONS: dict[str, Callable] = {
    "A": bench_build,
    "B": bench_exec_modes,
    "C": bench_logger,
    "D": bench_decompose,
    "E": bench_profile,
    "F": bench_scaling,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="超大规模框架开销基准测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
测试阶段:
  A  节点构建/哈希/广播开销
  B  串行 vs 并行执行对比
  C  Logger/Reporter 开销隔离
  D  逐环节 cProfile 分解
  E  cProfile Top-N 热点
  F  规模扩展趋势

示例:
  uv run python -m scripts.massive_overhead_benchmark
  uv run python -m scripts.massive_overhead_benchmark --dim-a 40 --dim-b 30 --dim-c 20 --depth 6
  uv run python -m scripts.massive_overhead_benchmark --section C D --profile
""",
    )
    parser.add_argument("--dim-a", type=int, default=30, help="维度 A 大小 (默认 30)")
    parser.add_argument("--dim-b", type=int, default=25, help="维度 B 大小 (默认 25)")
    parser.add_argument("--dim-c", type=int, default=20, help="维度 C 大小 (默认 20)")
    parser.add_argument("--depth", type=int, default=6, help="菱形依赖深度 (默认 6)")
    parser.add_argument(
        "--section", nargs="+", choices=list(SECTIONS),
        metavar="SECTION", help="仅运行指定阶段",
    )
    parser.add_argument("--profile", action="store_true", help="额外运行 cProfile 热点")
    args = parser.parse_args()

    da, db, dc, dep = args.dim_a, args.dim_b, args.dim_c, args.depth
    abc = da * db * dc
    est_total = abc * (1 + 3 * dep) + da * db + da + 1 + (da + db + dc)

    enabled = set(args.section) if args.section else set(SECTIONS) - {"E"}
    if args.profile:
        enabled.add("E")

    print(f"\n{_WIDE}")
    print(f"  超大规模框架开销基准测试")
    print(f"  dims={da}×{db}×{dc}={abc:,}  depth={dep}  预估节点≈{est_total:,}")
    print(f"  CPU={os.cpu_count()}  PID={os.getpid()}")
    print(f"{_WIDE}")

    for key in sorted(enabled):
        fn = SECTIONS[key]
        if key == "F":
            fn(dep)
        else:
            fn(da, db, dc, dep)

    print(f"\n{_WIDE}")
    print("  基准测试完成")
    print(f"{_WIDE}\n")


if __name__ == "__main__":
    main()
