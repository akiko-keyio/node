# 深依赖调度路径基准与瓶颈分析

## 目标

验证在「上万维度 + 深层依赖」时，Runtime 调度循环中以下过程的成本：

1. 处理已完成任务：`future.result`、`sem.release`、`ts.done`、依赖引用计数更新。
2. 发现可运行节点：`ts.get_ready`、`memory_aware_scheduling` 排序。
3. ready 前置动作：依赖参数解析、`cache.get`、`_ensure_deps_ready`。
4. 提交执行：`pool.submit`、`sem.acquire`。

## Bench 脚本

已新增：`scripts/deep_dependency_scheduler_benchmark.py`

核心特征：

- 维度规模可配置（默认 `12000`，建议 `>=10000`）。
- 依赖深度可配置（默认 `10`）。
- DAG 结构：每层 `add_bias` + `mul_scale` + `fuse`，并在末端 `reduce_dims="all"` 汇总。
- 输出：
  - `build_graph` 耗时
  - 执行耗时与吞吐（nodes/s）
  - `cProfile` 按调度阶段分组的 self-time 占比

## 运行命令

```bash
python -m scripts.deep_dependency_scheduler_benchmark --dim-size 10000 --depth 8 --workers 8 --memory-aware --repeats 1
python -m scripts.deep_dependency_scheduler_benchmark --dim-size 10000 --depth 8 --workers 8 --repeats 1
python -m scripts.deep_dependency_scheduler_benchmark --dim-size 10000 --depth 8 --workers 1 --repeats 1
```

## 实测结果（本机）

### Case A: `workers=8`, `memory_aware=True`

- 总节点数：`250,002`
- `build_graph`: `462.26 ms`
- 执行：`176,768.11 ms`
- 吞吐：`1,414 nodes/s`
- profile total self-time: `176,768.10 ms`
- 分组 self-time：
  - completed-task handling: `23,452.41 ms` (`13.3%`)
  - threadpool submission: `3,219.16 ms` (`1.8%`)
  - ready-node discovery: `305.51 ms` (`0.2%`)
  - ready pre-actions: `78.49 ms` (`<0.1%`)

### Case B: `workers=8`, `memory_aware=False`

- 总节点数：`250,002`
- `build_graph`: `544.96 ms`
- 执行：`160,699.51 ms`
- 吞吐：`1,556 nodes/s`
- profile total self-time: `160,699.49 ms`
- 分组 self-time：
  - completed-task handling: `24,020.49 ms` (`14.9%`)
  - threadpool submission: `3,607.73 ms` (`2.2%`)
  - ready-node discovery: `171.62 ms` (`0.1%`)
  - ready pre-actions: `95.71 ms` (`0.1%`)

### Case C: `workers=1`, `memory_aware=False`

- 总节点数：`250,002`
- `build_graph`: `629.56 ms`
- 执行：`7,843.98 ms`
- 吞吐：`31,872 nodes/s`
- profile total self-time: `7,843.97 ms`
- 分组 self-time：
  - ready pre-actions: `1,683.22 ms` (`21.5%`)
  - completed-task handling: `468.02 ms` (`6.0%`)
  - ready-node discovery: `0 ms`
  - threadpool submission: `0 ms`

## 结论（对应四段流程）

1. **处理已完成任务是并行路径中的第一类显著开销。**
   - 在 `workers=8` 下，约占 `13%~15%`。
   - 这是“高频小任务”场景的典型调度上限，任务本体很轻时尤为明显。

2. **发现可运行节点（`get_ready + 排序`）不是主要瓶颈。**
   - 占比仅 `0.1%~0.2%`。
   - `memory_aware` 会增加该项，但绝对值仍小。

3. **ready 前置动作在并行路径中占比较低；在串行路径中更显著。**
   - `workers=8`：约 `0.1%`。
   - `workers=1`：约 `21.5%`，因为没有线程池开销，参数解析/缓存检查被放大。

4. **提交到线程池是并行路径固定成本，且在细粒度任务下会主导退化。**
   - `workers=8` 约 `1.8%~2.2%`（按当前分组口径）。
   - 结合整体耗时可见：该 DAG 的单节点计算太轻，线程并发收益远小于调度成本。

## 关键观察

- 同一 DAG（25 万节点）：
  - `workers=1` 执行约 `7.84s`
  - `workers=8` 执行约 `160~177s`
- 说明此 workload 是 **典型“超细粒度节点”**：
  - 多线程并没有带来吞吐提升，
  - 反而被 `future` 生命周期、信号量、任务提交/回收等调度成本压垮。

## 优化建议（按收益优先级）

1. 对轻计算节点，优先使用 `workers=1` 或较小并发（如 `2`），避免线程池风暴。
2. 提升节点粒度（batch/fuse 多个小节点），减少 `submit/result/done` 次数。
3. 仅在“单节点计算足够重”时再提升 `workers`。
4. `memory_aware_scheduling` 建议按 DAG 形态开关；在该场景收益不明显。
