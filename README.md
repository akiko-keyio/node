# Node

Node 是一个轻量级的 Python DAG 执行框架，专为数据管线和计算密集型任务设计。它通过简洁的装饰器 API 自动处理**自动缓存**、**结果追溯**、**并行执行**、**配置管理**，让你专注于业务逻辑。

# 安装

```bash
pip install -e .
```

Python 版本需 ≥3.10。

# 快速开始

```python
import node

node.configure()

@node.define()
def add(x, y):
    return x + y

@node.define()
def square(n):
    return n * n

# 构建计算节点
node1 = square(add(2, 3))

# 执行计算
print(node1())  # 25
```

---

# 核心机制

## 自动缓存

计算结果自动缓存，相同输入不重复计算：

```python
import time

@node.define()
def slow_compute(x):
    time.sleep(2)
    return x * 2

result = slow_compute(5)()  # 首次：2秒
result = slow_compute(5)()  # 缓存命中：瞬间
result = slow_compute(6)()  # 参数变化：重新计算
```

### 缓存键

每个节点通过以下公式生成唯一标识：

```text
缓存键 = hash(函数名 + 参数 + 依赖节点的缓存键)
```

满足以下条件，相同缓存键生成相同结果：
- 相同函数名确定了相同的计算逻辑
- 所有影响结果的参数、依赖节点都被考虑（纯函数）

### 纯函数要求

```python
# ✅ 纯函数：相同输入 → 相同输出
@node.define()
def process(data, threshold):
    return [x for x in data if x > threshold]

# ❌ 闭包变量：factor 不在缓存键中
def make_processor(factor):
    @node.define()
    def process(x):
        return x * factor
    return process

# ❌ 非确定性输出
@node.define()
def generate(n):
    return [random.random() for _ in range(n)]
```

### 缓存后端

默认使用两级缓存：内存 LRU → 磁盘持久化。可通过 `node.configure()` 自定义：

```python
from node import ChainCache, MemoryLRU, DiskJoblib

node.configure(
    cache=ChainCache([
        MemoryLRU(maxsize=512),     # 热数据快速访问
        DiskJoblib(root=".cache"),  # 冷数据持久化
    ])
)
```

| 缓存类型     | 用途           | 配置项                                     |
| ------------ | -------------- | ------------------------------------------ |
| `MemoryLRU`  | 快速访问热数据 | `maxsize`: 最大条目数                      |
| `DiskJoblib` | 持久化到磁盘   | `root`: 缓存目录, `small_file`: 小文件阈值 |
| `ChainCache` | 组合多级缓存   | 按顺序查找，命中后回填                     |

---

## 结果追溯

缓存键是哈希值，不可逆。为支持**调试复现**，框架提供完整的计算脚本：

**方式一**：`repr(node)` 生成可执行脚本

```python
@node.define()
def A(): return 1

@node.define()
def B(a): return a + 1

root = B(A())
print(repr(root))
```

输出：

```python
# hash = 7c3b8fad541e11
A_0 = A()
B_0 = B(a=A_0)
```

**方式二**：磁盘缓存自动保存脚本

```text
.cache/
├── B/
│   ├── 7c3b8fad541e11.pkl    # 计算结果
│   └── 7c3b8fad541e11.py     # 复现脚本
```

运行 `.py` 文件即可复现完整计算过程。

---

## 任务调度

缓存命中后无需重新计算，但未命中的节点可以并行执行。框架调度策略如下：

1. 遍历 DAG，按依赖顺序收集所有节点
2. 检查每个节点的缓存状态
3. 已缓存：直接返回结果
4. 未缓存：提交到执行器（线程池/进程池）
5. 独立节点可并行执行

```python
node.configure(executor="thread", workers=4)

@node.define()
def download(url):
    return requests.get(url).text

@node.define()
def parse(html):
    return extract_data(html)

# d1, d2, d3 可并行下载
d1, d2, d3 = download(url1), download(url2), download(url3)

# 各自的 parse 也可并行（在对应 download 完成后）
results = gather([parse(d1), parse(d2), parse(d3)]).get()
```

### 执行器配置

```python
node.configure(
    executor="thread",   # "thread" 或 "process"
    workers=4,           # 默认并发数
)
```

**节点级控制：**

```python
@node.define(workers=2)      # 此函数最多 2 个并发
def heavy_task(): ...

@node.define(workers=-1)     # 使用全部 CPU 核心
def parallel_task(): ...

@node.define(local=True)     # 在主线程执行，绕过执行器
def io_task(): ...
```

## 配置管理

通过 YAML 集中管理节点默认参数：

```yaml
# config.yaml
data_path: "data/raw.csv"
threshold: 0.5

load_data:
  _target_: mymodule.load_data
  path: ${data_path}

process:
  _target_: mymodule.process #
  data: ${load_data}         # 自动创建依赖
  threshold: ${threshold}    # 引用全局配置
  _use_: development         # 预设
  _presets_:
    development: { threshold: 0.3 }
    production:  { threshold: 0.8 }
```

```python
from node import Config

node.configure(config=Config("config.yaml"))

@node.define()
def load_data(path):
    return pd.read_csv(path)

@node.define()
def process(data, threshold):
    return data[data.value > threshold]

# 配置自动注入
result = process().get()

# 切换预设
node.cfg.process._use_ = "production"
result = process().get()  # threshold=0.8
```

---

# 便利工具

## gather 聚合节点

`gather` 将多个独立节点合并为列表：

```python
from node import gather

@node.define()
def process(x):
    return x * 2

nodes = [process(i+1) for i in range(3)]
results = gather(nodes).get()
# → [2, 4, 6]
```

## sweep 参数扫描

`sweep` 通过遍历全局配置，生成多组独立的执行结果。配合 `${...}` 引用机制，只需调整顶级参数，即可自动更新整个依赖链所有节点的参数。

```python
from node import sweep, Config

# 配置依赖链：train -> preprocess -> mode
node.configure(config=Config({
    "mode": "fast",
    
    "preprocess": {
        "_target_": "__main__.preprocess", # 指定函数路径
        "method": "${mode}"
    },
    
    "train": {
        "data": "${preprocess}",       # 自动实例化并注入 preprocess 节点
        "strategy": "${mode}"
    }
}))

@node.define()
def preprocess(method):
    return f"Data({method})"

@node.define()
def train(data, strategy):
    return f"Result({data}, {strategy})"

# 扫描 mode，自动触发 preprocess 和 train 的重建
results = node.sweep(
    train,
    config={"mode": ["fast", "accurate"]}
)()

# ["Result(Data(fast), fast)", "Result(Data(accurate), accurate)"]
```

## track - 进度监控

在节点内部追踪循环进度：

```python
from node import track

@node.define()
def process_items(items):
    results = []
    for item in track(items, description="Processing"):
        results.append(compute(item))
    return results
```

进度条会显示在对应节点行下方，支持线程和进程池执行。
