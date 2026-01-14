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

根据节点标识（缓存键）维护计算结果，避免重复计算：

```python
import time

@node.define()
def slow_compute(x):
    time.sleep(2)
    return x * 2

# 第一次执行：计算并写入缓存（耗时2秒）
result = slow_compute(5)()

# 第二次执行：检测到相同的缓存键，直接返回结果（瞬间完成）
result = slow_compute(5)()

# 参数变化：重新计算
result = slow_compute(6)()  # 2秒
```

**节点标识**

缓存的关键在于如何唯一标识每个计算节点。节点标识定义为：

```text
节点标识 = hash(函数名 + 参数 + 依赖节点的标识)
```

如果满足以下条件，节点标识确定了唯一的输出结果：

- 相同函数名确定了相同的计算逻辑
- 所有影响结果的参数、依赖节点都被考虑（纯函数）
- 依赖节点的标识确定了唯一的依赖节点的结果

**纯函数要求**

为确保节点标识与输出结果的一一对应，所有使用 `@node.define` 装饰的函数必须是**纯函数**：

- 相同的输入永远产生相同的输出
- 不产生副作用（不修改外部状态、不执行 I/O）

```python
# ✅ 正确
@node.define()
def process(data, threshold):
    return [x for x in data if x > threshold]

# ❌ 错误：依赖闭包变量
def make_processor(factor):
    @node.define()
    def process(x):
        return x * factor  # factor 不在缓存键中
    return process

# ❌ 错误：非确定性
@node.define()
def generate(n):
    return [random.random() for _ in range(n)]
```

**缓存后端**

默认使用两级缓存：内存 LRU → 磁盘持久化。可通过 `node.configure()` 自定义：

```python
from node import ChainCache, MemoryLRU, DiskJoblib

node.configure(
    cache=ChainCache([
        MemoryLRU(maxsize=512),       # 内存缓存，LRU 淘汰
        DiskJoblib(root=".cache"),    # 磁盘缓存
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

缓存基于节点标识，但哈希值是不可逆的。为支持调试和复现，框架提供了两种方式查看完整的计算过程：

**方式一**：通过 `repr(node)` 获取可执行的 Python 脚本：

```python
a = A()
root = D(B(a), C(a))
print(repr(root))
```

```python
# hash = 7c3b8fad541e11
A_0 = A()
B_0 = B(a=A_0)
C_0 = C(a=A_0)
D_0 = D(b=B_0, c=C_0)
```

**方式二**：配置磁盘缓存 `DiskJoblib` 后，复现脚本自动保存在缓存目录：

```text
.cache/
├── D/
│   ├── 7c3b8fad541e11.pkl    # 计算结果
│   └── 7c3b8fad541e11.py     # 复现脚本（内容同上）
```

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

**执行器配置**

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
