# Node 框架设计文档

## 1. 任务定义

定义**计算节点 (Node)** 由三个要素完全决定：

1. **函数**：计算逻辑
2. **参数**：影响计算的所有参数
3. **依赖**：上游节点的计算结果

只要这三者相同，我们就认为这是同一个计算，应该得到相同的结果。

计算节点的组合构成数据管线，它们的依赖关系形成有向无环图（DAG）

---

## 2. 设计目标

框架旨在解决传统数据管线的以下问题

**1. 缓存难以管理**：

- 每个节点都要写重复的缓存逻辑（缓存键定义、缓存读写、缓存失效判断和处理、...）
- 缓存键要包含上游依赖，构造复杂

**2. 结果无法追溯**：

- 如果缓存键不完整，将无法识别计算结果的来源
- 即使缓存键完整，也无法追溯到计算具体执行代码

**3. 配置注入繁琐**：

- 大型管线配置多且难以管理
- 不同局部配置组的组合切换繁琐

---

## 3. 解决方案

### 快速开始

10 行代码体验核心特性：

```python
import node

node.configure()

@node.define
def add(x, y):
    return x + y

@node.define
def square(n):
    return n * n

# 构建 DAG 并执行
result = square(add(2, 3)).get()
print(result)  # 25

# 再次执行 → 直接从缓存返回
result = square(add(2, 3)).get()

# 查看执行脚本
print(repr(square(add(2, 3))))
# add_0 = add(x=2, y=3)
# square_0 = square(n=add_0)
```

---

### 3.1 自动缓存（解决问题1）

#### 基本用法

用 `@node.define` 装饰函数，框架自动处理缓存：

```python
import node

@node.define
def load_data(path: str):
    return pd.read_csv(path)

@node.define
def process(data, threshold: float):
    return data[data.value > threshold]

result = process(load_data("data.csv"), threshold=0.5).get()
```

#### 缓存配置

默认使用两级缓存：内存 LRU → 磁盘持久化。可自定义：

```python
from node import ChainCache, MemoryLRU, DiskJoblib

node.configure(
    cache=ChainCache([
        MemoryLRU(maxsize=512),          # 内存缓存，LRU 淘汰
        DiskJoblib(root=".cache"),       # 磁盘缓存
    ])
)
```

| 缓存类型     | 用途           | 配置项                                                                               |
| ------------ | -------------- | ------------------------------------------------------------------------------------ |
| `MemoryLRU`  | 快速访问热数据 | `maxsize`: 最大条目数                                                                |
| `DiskJoblib` | 持久化到磁盘   | `root`: 缓存目录<br>`small_file`: 小文件阈值（默认 1MB，低于此值用 pickle 加载更快） |
| `ChainCache` | 组合多级缓存   | 按顺序查找，命中后回填到更快的缓存                                                   |

#### 缓存失效

参数或上游任何变化自动触发重新计算：

```python
# 第一次运行：执行计算并缓存
result = process(load_data("data.csv"), threshold=0.5).get()

# 第二次运行：直接从缓存返回
result = process(load_data("data.csv"), threshold=0.5).get()

# 参数变化 → 缓存失效 → 重新计算
result = process(load_data("data.csv"), threshold=0.8).get()
```

#### 禁用/忽略缓存

```python
# 禁用此节点的缓存
@node.define(cache=False)
def volatile_task(): ...

# 从缓存键中排除某些参数（这些参数不影响输出）
@node.define(ignore=["logger", "profiler"])
def train(data, logger): ...
```

---

### 3.2 可追溯的执行记录（解决问题2）

#### 脚本生成

`repr(node)` 返回可执行的 Python 脚本，展示该节点及其所有依赖的完整计算过程：

```python
root = D(B(A()), C(A()))
print(repr(root))
```

输出：

```python
# hash = 7c3b8fad541e11
A_0 = A()
B_0 = B(a=A_0)
C_0 = C(a=A_0)
D_0 = D(b=B_0, c=C_0)
```

**特点**：
- 每个节点有唯一的 hash 标识
- 变量名自动生成（`函数名_序号`）
- 菱形依赖正确处理：A 只出现一次，B 和 C 都引用同一个 `A_0`
- 可直接复制执行，复现计算结果

#### 磁盘脚本

`DiskJoblib` 自动保存脚本到缓存目录：

```
.cache/
├── A/
│   ├── abc123.pkl    # 计算结果
│   └── abc123.py     # 生成脚本
├── B/
│   └── ...
└── D/
    └── 7c3b8f.py     # 完整 DAG 脚本
```

**用途**：复现历史结果、审计计算过程、调试问题。

---

### 3.3 配置系统（解决问题3）

#### 基本配置

通过 YAML 集中管理节点默认参数：

```yaml
# config.yaml
data_path: "data.csv"
threshold: 0.5

process:
  _target_: mymodule.process   # 指定函数路径
  threshold: ${threshold}       # 引用其他配置值
```

```python
import node
from node import Config

node.configure(config=Config("config.yaml"))

@node.define
def process(data, threshold):
    return data[data.value > threshold]

# threshold 自动从配置注入
result = process(data).get()
```

#### 节点引用

配置可以引用其他节点，自动构建依赖：

```yaml
load_data:
  _target_: mymodule.load_data
  path: "data.csv"

process:
  _target_: mymodule.process
  data: ${load_data}            # 自动创建 load_data 节点并注入
  threshold: 0.5
```

```python
# 无需手动调用 load_data()
result = process().get()   # data 参数自动注入为 load_data 节点
```

#### 配置预设

当同一节点需要多套配置时，使用预设：

```yaml
process:
  _target_: mymodule.process
  threshold: 0.5                 # 基础配置
  _use_: normal                  # 当前选中的预设
  _presets_:
    normal:  { threshold: 0.5 }
    strict:  { threshold: 0.9 }
    relaxed: { threshold: 0.1 }
```

**解析规则**：`最终配置 = 基础配置 + 选中预设（预设覆盖基础）`

**运行时切换**：

```python
# 使用默认预设 (normal)
result = process(data).get()

# 切换到 strict 预设
node.cfg.process._use_ = "strict"
result = process(data).get()  # threshold=0.9
```

---

### 3.4 调度运行

#### 执行器配置

```python
node.configure(
    executor="thread",   # "thread" 或 "process"
    workers=4,           # 默认并发数
)
```

#### 节点级并发控制

```python
@node.define(workers=2)      # 此函数最多 2 个并发
def heavy_task(): ...

@node.define(workers=-1)     # 使用全部 CPU 核心
def parallel_task(): ...

@node.define(local=True)     # 在主线程执行，绕过执行器
def io_task(): ...
```

#### 智能调度

框架自动处理菱形依赖，避免重复计算：

```text
    A
   / \
  B   C
   \ /
    D
```

```python
@node.define
def A(): return expensive_computation()

@node.define
def B(a): return transform1(a)

@node.define
def C(a): return transform2(a)

@node.define
def D(b, c): return combine(b, c)

result = D(B(A()), C(A())).get()
```

**执行过程**：

1. A 执行一次，结果被 B 和 C 共享
2. B、C 可并行执行
3. D 等待 B、C 完成后执行

框架**不会重复计算 A**，即使它被 B 和 C 同时依赖。

---

### 3.5 实用函数

#### gather：聚合多个节点

```python
from node import gather

# 并行执行多个节点，结果合并为列表
result = gather([
    process(data1),
    process(data2),
    process(data3),
], workers=3).get()
# → [result1, result2, result3]
```

适用于聚合**不同函数**或**不同参数组合**的节点。

---

### 3.6 进度显示

使用 `RichReporter` 显示实时执行进度：

```python
from node import RichReporter

result = square(add(2, 3)).get()  # 自动使用默认 reporter
```

**进度显示效果**：

```text
┌────────────────────────────────────────┐
│ Node Execution                         │
├────────────────────────────────────────┤
│ ✔ add_a1b2c3        0.01s               │
│ ✔ square_d4e5f6     0.00s  [cached]     │
│ ▶ process_789abc    [███████   ] 70%   │
│   ┗ Processing: 7/10 items              │
└────────────────────────────────────────┘
```

#### 节点内进度条

使用 `track()` 在节点内部追踪循环进度：

```python
from node import track

@node.define
def process(items):
    results = []
    for item in track(items, description="Processing"):
        results.append(compute(item))
    return results
```

进度条会显示在对应节点行下方，支持线程和进程池执行。
