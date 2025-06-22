# Node

Node 是一个轻量级、零依赖的 DAG 流程库，适合在脚本或小型项目中快速组织多步计算。它通过兩級缓存加速执行，并提供直观的脚本表示，方便调试和分享。

## 功能特性

- **任务装饰器**：普通函数经 `@flow.node()` 包装后即可组成 DAG，可用 `ignore` 参数排除大型对象。
- **两级缓存**：默认同时启用内存 LRU 与磁盘缓存，避免重复计算。
- **并行执行**：支持线程或进程池，可在装饰器中用 `workers` 指定单个任务的并发数。
- **脚本表示**：任意节点的 `repr()` 都会生成等效的 Python 调用脚本。
- **配置系统**：通过 `Config` 对象集中管理任务默认参数，支持从 YAML 加载。
- **回调钩子**：`on_node_end` 与 `on_flow_end` 可用来收集统计信息。
- **结果聚合**：`gather` 工具可将多个节点合并为一个列表返回，方便并行处理。
- **日志模块**：`from node import logger` 即可获得预配置的 `loguru` 记录器。

## 安装

推荐在虚拟环境中执行：

```bash
pip install -e .
```

项目依赖见 `pyproject.toml`，Python 版本需 >=3.10。

## 快速开始

```python
from node.node import Flow, gather

flow = Flow()


@flow.node(workers=2)
def add(x, y):
    return x + y


@flow.node()
def square(z):
    return z * z

@flow.node(ignore=["large_df", "model"])
def train(x, y, large_df, model):
    return x + y


result = flow.run(square(add(2, 3)))
print(result)  # 25

# gather 多个独立节点（也可传入列表）
result = flow.run(gather([add(1, 2), add(3, 4)]))
print(result)  # [3, 7]
```

## 查看脚本

节点的 `repr` 形式即完整的执行脚本，便于复制和排查：

```python
root = square(add(2, 3))
print(repr(root))
# add_89f9dde66a325c89ee8739040c6147ad = add(x=2, y=3)
# square_7c3b8fad541e116101dc48cc17f9707b = square(z=add_89f9dde66a325c89ee8739040c6147ad)
```

## 缓存与并行

`Flow` 默认使用 `MemoryLRU` 和 `DiskJoblib` 组合成 `ChainCache`。缓存键为 `"<func>_<digest>"` 格式的哈希值，磁盘缓存统一存放在 `<func>/<digest>.pkl`，同时写入 `<digest>.py` 保存脚本：

```python
from node.node import Flow, ChainCache, MemoryLRU, DiskJoblib

flow = Flow(
    cache=ChainCache([MemoryLRU(), DiskJoblib(".cache")]),
    executor="thread",  # 或 "process"
    default_workers=4,
)

```

`DiskJoblib` 提供 `small_file` 参数，当缓存文件体积在该阈值内时，加载阶段将直接调用 `pickle.load`，避免 `joblib` 在海量小文件场景下的开销。

在 500 个 6 字节字符串的基准测试中，设置 `small_file=1_000_000` 的第二次读取耗时约
`0.0002` 秒，而强制使用 `joblib` 时约 `0.003` 秒，速度提升近 15 倍。基准脚本位于
`tests/test_small_file_perf.py`。
运行 `tutorial.py` 后将在缓存目录生成以下文件：

```
.cache/inc/89f9dde66a325c89ee8739040c6147ad.pkl
.cache/inc/89f9dde66a325c89ee8739040c6147ad.py
```

## 配置对象

使用 `Config` 管理任务默认参数：

```python
import yaml
from node.node import Flow, Config

with open("defaults.yml") as f:
    defaults = yaml.safe_load(f)

flow = Flow(config=Config(defaults))
```

示例 `defaults.yml`：

```yaml
add:
  y: 5
```

测试目录中的 `tests/config.yaml` 亦可参考。

## 教程脚本

`tutorial.py` 演示了完整工作流，包括：

1. 构建并运行 DAG；
2. 从 YAML 加载默认参数；
3. 查看磁盘缓存文件及生成的 `.py` 脚本。

运行：

```bash
python tutorial.py
```

## 解决多进程下 RichReporter 跳动

在进程池执行时，多个进程同时向终端写入会导致 `RichReporter` 的进度条跳动。
可以在主进程独占终端，并强制 Rich 将控制台视为真实终端：

```python
from rich.console import Console
from node.reporters import RichReporter

flow = Flow(executor="process")
reporter = RichReporter(console=Console(force_terminal=True))
flow.run(root, reporter=reporter)
```

确保工作进程不要向标准输出打印内容，以避免刷新冲突。

## 节点内部进度条

`track` 辅助函数可用于在节点内部追踪循环进度，并与 `RichReporter` 的
实时界面相结合：

```python
from node import track

@flow.node()
def consume(items):
    total = 0
    for x in track(items, description="Processing", total=len(items)):
        total += x
    return total
```

运行时，进度条会显示在对应节点的行下方，不会影响其它节点的展示。
在进程池执行时亦可使用 ``track``，进度信息会通过主进程统一渲染，避免多进程输出冲突。




