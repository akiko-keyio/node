# Node

Node 是一个轻量级、零依赖的 DAG 流程库，适合在脚本或小型项目中快速组织多步计算。它通过兩級缓存加速执行，并提供直观的脚本表示，方便调试和分享。

## 功能特性

- **任务装饰器**：普通函数经 `@flow.node()` 包装后即可组成 DAG，可用 `ignore` 参数排除大型对象。
- **两级缓存**：默认同时启用内存 LRU 与磁盘缓存，避免重复计算。
- **并行执行**：支持线程或进程池，`workers` 参数控制并发量。
- **脚本表示**：任意节点的 `repr()` 都会生成等效的 Python 调用脚本。
- **配置系统**：通过 `Config` 对象集中管理任务默认参数，支持从 YAML 加载。
- **回调钩子**：`on_node_end` 与 `on_flow_end` 可用来收集日志或统计信息。

## 安装

推荐在虚拟环境中执行：

```bash
pip install -e .
```

项目依赖见 `pyproject.toml`，Python 版本需 >=3.10。

## 快速开始

```python
from node.node import Flow

flow = Flow()


@flow.node()
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
```

## 查看脚本

节点的 `repr` 形式即完整的执行脚本，便于复制和排查：

```python
root = square(add(2, 3))
print(repr(root))
# square(z=add(x=2, y=3))
```

## 缓存与并行

`Flow` 默认使用 `MemoryLRU` 和 `DiskJoblib` 组合成 `ChainCache`。缓存键即节点的 `signature` 字符串，磁盘缓存会以函数名创建子目录并尝试使用表达式作为文件名。这里的 `signature` 指节点的唯一字符串标识，由函数名及其参数构成，例如 `add(x=2, y=3)`：

```python
from node.node import Flow, ChainCache, MemoryLRU, DiskJoblib

flow = Flow(
    cache=ChainCache([MemoryLRU(), DiskJoblib(".cache")]),
    executor="thread",  # 或 "process"
    workers=4,
)
```

若表达式不适合作文件名，会回退到哈希值，并在同目录生成 `.py` 文件记录脚本。例如运行 `tutorial.py` 后可能出现：

```
.cache/inc/inc(x=3).pkl
.cache/add/8be18ab9a193dbbc86b394bac923ab03.pkl
.cache/add/8be18ab9a193dbbc86b394bac923ab03.py
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

日志会展示每个任务的执行时间与缓存命中情况。
此外，默认开启的 Rich 进度表会在节点开始执行时将其加入列表，并实时更新状态（运行中、已完成或来自缓存），同时把日志显示在进度表下方。

