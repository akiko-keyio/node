# Node

Node 是一个轻量级的 Python DAG 执行引擎，专为脚本和小型项目设计。它通过简洁的装饰器 API 将普通函数组织成有向无环图，并提供**两级缓存**、**并行执行**和**实时进度显示**等功能。

## 功能特性

- **模块级 API**：`import node` 即可使用 `@node.define` 装饰器和 `node.run()` 执行函数
- **两级缓存**：内存 LRU + 磁盘持久化，避免重复计算
- **并行执行**：线程池或进程池执行器，支持节点级并发控制
- **脚本表示**：任意节点的 `repr()` 生成等效 Python 脚本，便于调试和复现
- **配置系统**：通过 YAML 管理任务默认参数，支持 `${...}` 引用语法
- **实时进度**：`RichReporter` 提供实时任务状态和进度条
- **节点内进度**：`track()` 函数在节点内追踪循环进度

## 安装

```bash
pip install -e .
```

Python 版本需 ≥3.10。

## 快速开始

```python
import node

# 可选：配置运行时（不调用则使用默认值）
node.configure(workers=4, executor="thread")

@node.define
def add(x, y):
    return x + y

@node.define
def square(z):
    return z * z

# 构建 DAG 并执行
result = node.run(square(add(2, 3)))
print(result)  # 25
```

## 设计约束

### 纯函数要求

**重要**：所有使用 `@node.define` 装饰的函数必须是**纯函数**。

**纯函数定义**：
- 相同的输入永远产生相同的输出（确定性）
- 不产生副作用（不修改外部状态、不执行 I/O）

**正确示例** ✅：
```python
@node.define
def process(data, threshold):
    return [x for x in data if x > threshold]
```

**错误示例** ❌：
```python
# ❌ 依赖闭包变量（框架会发出警告）
def make_processor(factor):
    @node.define
    def process(x):
        return x * factor  # factor 不在缓存键中
    return process

# ❌ 非确定性输出
import random
@node.define
def generate(n):
    return [random.random() for _ in range(n)]

# ❌ 副作用
@node.define
def save_and_process(data):
    with open("output.txt", "w") as f:
        f.write(str(data))  # 副作用
    return data * 2
```

**为什么需要纯函数**：框架依赖纯函数性保证缓存正确性、支持并行执行和结果可重现性。

## 装饰器参数

`@node.define` 支持以下参数：

```python
@node.define(
    ignore=["large_df"],  # 从缓存键中排除的参数名
    workers=2,            # 此函数的最大并发数，-1 表示使用全部核心
    cache=True,           # 是否缓存结果
    local=True,           # 在主线程执行，绕过执行器
)
def train(x, y, large_df):
    return x + y
```

> **关于 `ignore` 参数**：被忽略的参数**必须真正不影响函数输出**。框架不会将这些参数包含在缓存键中，
> 如果它们实际影响了结果，会导致缓存返回错误的值。
>
> **正确用法**：日志器、性能分析器、预分配缓冲区等不影响计算逻辑的对象。
>
> **错误用法**：随机种子、数据源路径等任何影响输出的参数。

## 运行时配置

使用 `node.configure()` 配置全局运行时：

```python
import node
from node import ChainCache, MemoryLRU, DiskJoblib, Config

node.configure(
    config=Config("defaults.yml"),  # 配置对象或 YAML 路径
    cache=ChainCache([MemoryLRU(), DiskJoblib(".cache")]),
    executor="thread",  # "thread" 或 "process"
    workers=4,          # 默认并发数
)
```

> **注意**：`node.configure()` 只能调用一次。如需重置，使用 `node.reset()`。

## 节点操作

每个节点对象提供以下方法：

```python
root = square(add(2, 3))

# 执行并返回结果（等同于 node.run(root)）
result = root.get()

# 删除此节点的缓存
root.delete()

# 强制重新计算，忽略现有缓存
result = root.create()

# 计算并缓存，但不返回值（预热）
root.generate()
```

## 聚合多个节点

使用 `gather` 将多个独立节点合并为列表：

```python
from node import gather

result = node.run(gather([add(1, 2), add(3, 4)], workers=2))
print(result)  # [3, 7]
```

## 批量映射

使用 `map` 将节点函数批量应用于多个参数：

```python
from node import map as node_map

# 单参数映射
result = node.run(node_map(process, x=[1, 2, 3]))
# 等价于: gather([process(x=1), process(x=2), process(x=3)])

# 多参数映射（zip 行为）
result = node.run(node_map(compare, a=[1, 2], b=[3, 4]))
# 等价于: gather([compare(a=1, b=3), compare(a=2, b=4)])

# 参数值为列表时，需要嵌套列表
result = node.run(node_map(process_batch, items=[[1, 2], [3, 4]]))
# 创建: process_batch(items=[1,2]), process_batch(items=[3,4])
```

## 脚本表示

节点的 `repr()` 生成完整的执行脚本，便于复制和排查：

```python
root = square(add(2, 3))
print(repr(root))
# add_89f9dde66a325c89ee8739040c6147ad = add(x=2, y=3)
# square_7c3b8fad541e116101dc48cc17f9707b = square(z=add_89f9dde66a325c89ee8739040c6147ad)
```

## 缓存系统

默认使用 `ChainCache([MemoryLRU(), DiskJoblib(".cache")])` 组合缓存。

### 磁盘缓存结构

```
.cache/
├── add/
│   ├── 89f9dde66a325c89ee8739040c6147ad.pkl  # 计算结果
│   └── 89f9dde66a325c89ee8739040c6147ad.py   # 生成脚本
└── square/
    └── ...
```

### 小文件优化

`DiskJoblib` 的 `small_file` 参数（默认 1MB）指定阈值：小于该值的文件使用 `pickle` 加载，避免 `joblib` 在大量小文件场景下的开销。

## 配置文件

使用 YAML 配置任务默认参数：

```yaml
# defaults.yml
base:
  val: 5
add:
  y: ${base.val}  # 引用其他配置
```

```python
import node
from node import Config

node.configure(config=Config("defaults.yml"))
```

访问和修改配置：

```python
runtime = node.get_runtime()
cfg = runtime.config._conf
cfg.add.y = 3
runtime.reset_config()  # 恢复初始配置
```

## 实时进度显示

使用 `RichReporter` 显示任务执行进度：

```python
from node import RichReporter

result = node.run(root, reporter=RichReporter())
```

### 多进程场景

在进程池执行时，强制 Rich 将控制台视为终端：

```python
from rich.console import Console
from node import RichReporter

node.configure(executor="process")
reporter = RichReporter(console=Console(force_terminal=True))
node.run(root, reporter=reporter)
```

## 节点内进度条

`track()` 函数在节点内部追踪循环进度：

```python
from node import track

@node.define
def process(items):
    total = 0
    for x in track(items, description="Processing", total=len(items)):
        total += x
    return total
```

进度条会显示在对应节点行下方，支持线程和进程池执行。

## Runtime 类（高级用法）

对于需要多个独立运行时的场景，可直接使用 `Runtime` 类：

```python
from node import Runtime, ChainCache, MemoryLRU, DiskJoblib

runtime = Runtime(
    cache=ChainCache([MemoryLRU(), DiskJoblib(".cache")]),
    executor="thread",
    workers=4,
)

@runtime.define()
def add(x, y):
    return x + y

result = runtime.run(add(1, 2))
```

> **向后兼容**：`Flow` 是 `Runtime` 的别名。

## 教程脚本

运行 `tutorial.py` 查看完整示例：

```bash
python tutorial.py
```

## 日志

内置 `loguru` 日志记录器：

```python
from node import logger

logger.info("Processing started")
```
