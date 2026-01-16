# Node

Node 是一个轻量级的 Python DAG 执行框架，专为数据管线和计算密集型任务设计。通过装饰器 API 自动处理**缓存**、**追溯**、**并行**、**配置**。

## 安装

```bash
uv pip install -e .
```

Python ≥3.10

## 快速开始

```python
import node

node.configure()  # 初始化

@node.define() # 定义节点
def add(x, y):
    return x + y

@node.define()
def square(n):
    return n * n

result = square(add(2, 3))  # 构建 DAG，不执行计算
print(result())  # 执行并输出 25
```

---

# 核心机制

## 缓存

框架自动缓存计算结果。第二次执行时，如果输入不变直接返回缓存结果：

```python
@node.define()
def slow_compute(x):
    time.sleep(2)
    return x * 2

slow_compute(5)()  # 首次执行：2秒
slow_compute(5)()  # 缓存命中：瞬间返回
slow_compute(6)()  # 参数变化：重新计算
```

**缓存键计算**

每个节点缓存键由以下因素决定：

```
当前节点缓存键 = hash(函数名 + 参数值 + 依赖节点的缓存键)
```

**函数名**

函数名标识计算逻辑，函数体变更不会自动使缓存失效。修改函数实现后需手动清理

```python
result.invalidate()  # 清除单个节点的缓存
```

**纯函数要求**

为了保证缓存正确性，节点函数必须是纯函数（相同输入产生相同输出）：

```python
# ✓ 纯函数：相同输入 → 相同输出
@node.define()
def process(data, threshold):
    return [x for x in data if x > threshold]

# ✗ 闭包变量不在缓存键中，可能导致缓存不一致
def make_processor(factor):
    @node.define()
    def process(x):
        return x * factor  # factor 变化不会触发重算！
    return process

# ✗ 非确定性输出，缓存结果与实际执行结果不一致
@node.define()
def random_data(n):
    return [random.random() for _ in range(n)]
```

**禁用缓存**

对于确实需要每次执行的函数（如获取实时数据），可以禁用缓存：

```python
@node.define(cache=False)
def get_current_time():
    return datetime.now()
```

**排除参数**

某些参数（如调试标志）不应影响缓存键。使用 `ignore` 排除：

```python
@node.define(ignore=["debug", "verbose"]) # `debug` 和 `verbose` 变化不影响缓存键
def compute(x, debug=False, verbose=False):
    if debug: print(f"Computing {x}")
    return x * 2
```


**缓存后端**

默认使用两级缓存：内存 LRU → 磁盘。可自定义缓存策略：

```python
from node import ChainCache, MemoryLRU, DiskJoblib

node.configure(cache=ChainCache([
    MemoryLRU(maxsize=512),      # 内存中保留最近 512 个结果
    DiskJoblib(root=".cache"),   # 持久化到磁盘
]))
```

| 类型         | 用途           | 配置                           |
| ------------ | -------------- | ------------------------------ |
| `MemoryLRU`  | 热数据快速访问 | `maxsize`: 最大条目数          |
| `DiskJoblib` | 冷数据持久化   | `root`: 缓存目录               |
| `ChainCache` | 多级组合       | 按顺序查找，命中后回填上级缓存 |

---

## 追溯

缓存键是哈希值，不可逆。为支持调试复现，框架提供计算脚本生成功能。

**通过 repr 生成脚本**

对任意节点调用 `repr()` 可获得可执行的 Python 脚本：

```python
print(repr(square(add(2, 3))))
```


```python
# hash = 7c3b8fad541e11
add_0 = add(x=2, y=3)
square_0 = square(z=add_0)
```

**磁盘缓存自动保存脚本**

使用 `DiskJoblib` 缓存时，脚本会自动保存在缓存目录：

```
.cache/
├── square/
│   ├── 7c3b8fad541e11.pkl    # 计算结果
│   └── 7c3b8fad541e11.py     # 复现脚本
```

---

## 调度

未缓存的节点可并行执行。框架会自动分析 DAG 依赖关系，将无依赖的节点同时提交到执行器：

```python
node.configure(executor="thread", workers=4)

@node.define()
def download(url):
    return requests.get(url).text

# d1, d2, d3 没有相互依赖，可以并行执行
d1, d2, d3 = download(url1), download(url2), download(url3)
```

**执行器类型**

| 类型        | 适用场景                             |
| ----------- | ------------------------------------ |
| `"thread"`  | I/O 密集型任务（网络请求、文件读写） |
| `"process"` | CPU 密集型任务（计算、数据处理）     |

**节点级并发控制**

可以为特定函数设置并发限制：

| 配置         | 行为                                   |
| ------------ | -------------------------------------- |
| 不指定       | 使用 `configure(workers=N)` 的全局设置 |
| `workers=2`  | 该函数最多 2 个实例同时运行            |
| `workers=-1` | 使用全部 CPU 核心                      |
| `local=True` | 在主线程执行，不进入线程池             |

```python
@node.define(workers=2)
def heavy_task(x):
    # 资源密集型任务，限制并发
    return expensive_computation(x)

@node.define(local=True)
def gui_task():
    # 需要访问主线程资源（如 GUI 组件）
    update_ui()
```

---

## 错误处理

默认情况下，单个节点失败不会阻止其他节点执行。框架会跳过依赖失败节点的下游节点，继续执行其他分支：

```python
node.configure(continue_on_error=True)  # 默认值

@node.define()
def may_fail(x):
    if x < 0:
        raise ValueError("x must be positive")
    return x

@node.define()
def downstream(a, b):
    return a + b

# a 失败不影响 b 执行，但 downstream 会被跳过
a = may_fail(-1)
b = may_fail(1)
result = downstream(a, b)
```

设置 `continue_on_error=False` 可在首个节点失败时立即终止整个 DAG 执行。


---

## 配置

使用 YAML 文件集中管理节点的默认参数和依赖节点

**加载配置**

```python
from node import Config

node.configure(config=Config("config.yaml"))
```

```yaml
# config.yaml
load_data:
  path: "data.csv"
  threshold: 0.5
```

**注入默认参数**

配置中与函数同名的节会自动注入为该函数的默认参数：

```python
@node.define()
def load_data(path, threshold):
    df = pd.read_csv(path)
    return df[df.value > threshold]

# 不传参，使用配置值
result = load_data()()

# 调用时传参覆盖配置
result = load_data(path="other.csv")()
```

**修改默认参数**

通过 `node.cfg` 可以在运行时动态修改配置：

```python
node.cfg.load_data.path = "custom.csv"
load_data()()  # 使用修改后的值
```


**变量引用**

使用 `${...}` 引用其他配置项：

```yaml
base_dir: "/data"
raw_path: "${base_dir}/raw.csv"       # → "/data/raw.csv"
processed_path: "${base_dir}/out.csv" # → "/data/out.csv"
```

**节点依赖**

当依赖的前置节点包含 `_target_` 关键字时，框架根据 `_target_` 指定的函数路径自动构建节点并注入：

```yaml
load_data:
  _target_: mymodule.load_data  # 指定函数路径
  path: "data.csv"

process:
  _target_: mymodule.process
  data: ${load_data}  # 自动构建 load_data 节点并注入
  threshold: 0.5
```


**预设（Presets）**

为同一节点定义多套参数配置，运行时切换：

```yaml
train:
  learning_rate: 0.01
  epochs: 100
  _use_: dev           # 当前使用的预设
  _presets_:
    dev:  { epochs: 10 }       # 开发：快速迭代
    prod: { epochs: 1000 }     # 生产：完整训练
```

```python
train()()  # epochs=10（使用 dev 预设）

node.cfg.train._use_ = "prod"
train()()  # epochs=1000（切换到 prod 预设）
```

---

# 多维计算

多维计算遍历参数的多个取值，自动执行所有组合。


## 维度/坐标


通过 `@node.dimension()` 定义维度和坐标
- 维度是待遍历的参数名，维度名默认为函数名
- 坐标是参数的取值，为函数返回值列表

```python
@node.dimension()
def time():
    return [2020, 2021, 2022]
```

---

## 广播


当节点输入包含维度时，框架会自动广播计算

```python
@node.define()
def load(t):
    return pd.read_csv(f"{t}.csv")

# 对不同时间维度执行 3 次 load
data = load(t=time())() 
```

**多维广播**

不同维度的输入会自动计算笛卡尔积：

```python
@node.dimension()
def model():
    return ["LR", "RF"]

@node.define()
def train(t, m):
    return fit(m, t)

# 3 个时间 × 2 个模型 = 6 个独立计算
grid = train(t=time(), m=model())
```

**维度对齐**

当多个输入共享相同维度时，框架会自动对齐（zip），而非计算笛卡尔积：

```python
t = time()
s1 = step1(t=t)            
s2 = step2(x=s1, t=t)     

s2() # step1 和 step2 共享 "time" 维度，对齐处理
```

---

## 结果访问

执行多维计算节点后返回 `DimensionedResult`，它是 `numpy.ndarray` 的子类，额外携带维度信息：

```python
result = load(t=time())()
```

**属性**

```python
result.dims    # ("time",) - 维度名元组
result.coords  # {"time": [2020, 2021, 2022]} - 坐标字典
result.shape   # (3,) - 数组形状
```

**索引和切片**

```python
result[0]      # 第一个元素（2020 年的数据）
result[-1]     # 最后一个元素（2022 年的数据）
result[1:]     # 切片
```

**按坐标查找**

```python
idx = result.coords["time"].index(2021)
result[idx]    # 2021 年的数据
```

**多维操作**

```python
grid = train(t=time(), m=model())()
# grid.dims = ("model", "time"), grid.shape = (2, 3)

grid[0, 1]                        # 单元素：model=LR, time=2021
grid[1, :]                        # 切片：model=RF 的所有时间
grid.transpose("time", "model")   # 转置轴顺序
```

**带坐标遍历**

`items()` 方法遍历 `(元素, 坐标字典)` 对：

```python
# 合并带坐标的 DataFrame
dfs = []
for df, coords in result.items():
    df = df.copy()
    df["year"] = coords["time"]
    dfs.append(df)
final = pd.concat(dfs)
```

---

## 聚合

聚合多维计算结果，计算平均值、找最优、生成汇总报告等。


**全归约**

使用 `reduce_dims` 声明要归约的维度，默认不归约，设置 `"all"` 归约所有维度。

```python
@node.define(reduce_dims="all")
def summary(data: DimensionedResult) -> dict:
    # 函数接收完整 DimensionedResult
    # 返回标量结果
    return {
        "count": len(data.flat),
        "total": sum(d["value"] for d in data.flat)
    }
```

**部分归约**

归约指定维度，保留其他维度：

```python
@node.dimension()
def site():
    return ["A", "B"]

@node.define()
def measure(t, s):
    return {"time": t, "site": s, "value": t}

@node.define(reduce_dims=["time"])  # 声明归约 time 维度
def site_avg(data):
    # 输入: 包含某个 site 的所有 time 维度数据
    # 输出: 该 site 的平均值
    return sum(d["value"] for d in data.flat) / len(data.flat)

grid = measure(t=time(), s=site())  # dims=("site", "time"), shape=(2, 3)
avgs = site_avg(data=grid)          # dims=("site",), shape=(2,)
```


---

## 完整示例

以下是一个完整的数据管线示例，展示维度定义、广播和聚合的配合使用：

```python
import node
import numpy as np

node.configure()

# 定义维度
@node.dimension()
def time():
    return [2020, 2021, 2022]

@node.dimension()
def model():
    return ["LR", "RF"]

# 数据加载：按时间广播
@node.define()
def load(t):
    np.random.seed(t)  # 确保可复现
    return np.random.randn(100)

# 模型训练：按时间×模型广播
@node.define()
def train(data, m):
    score = np.mean(data) + (0.1 if m == "RF" else 0)
    return {"model": m, "score": score}

# 结果汇总：全维度归约
@node.define(reduce_dims="all")
def report(data):
    scores = [d["score"] for d in data.flat]
    return {
        "best_score": max(scores),
        "avg_score": sum(scores) / len(scores),
        "model_count": len(scores)
    }

# 构建并执行
times = time()
models = model()
result = report(data=train(data=load(t=times), m=models))()

print(result)
# {"best_score": ..., "avg_score": ..., "model_count": 6}

# 访问中间结果
trained = train(data=load(t=times), m=models)()
print(f"训练结果维度: {trained.dims}")  # ("model", "time")

for item, coords in trained.items():
    print(f"  {coords['model']}-{coords['time']}: {item['score']:.3f}")
```

---

# API 参考

## node.configure()

初始化运行时环境。必须在使用任何节点前调用，且只能调用一次。

| 参数                | 默认值                                    | 说明                                  |
| ------------------- | ----------------------------------------- | ------------------------------------- |
| `config`            | `None`                                    | Config 对象或 YAML 文件路径           |
| `cache`             | `ChainCache([MemoryLRU(), DiskJoblib()])` | 缓存后端                              |
| `executor`          | `"thread"`                                | 执行器类型：`"thread"` 或 `"process"` |
| `workers`           | `4`                                       | 默认并发数                            |
| `continue_on_error` | `True`                                    | 节点失败时是否继续执行其他节点        |
| `validate`          | `True`                                    | 是否使用 Pydantic 验证参数类型        |

```python
# 重新配置需先重置
node.reset()
node.configure(workers=8)
```

## @node.define()

将函数转换为节点工厂。

| 参数          | 默认值   | 说明                              |
| ------------- | -------- | --------------------------------- |
| `cache`       | `True`   | 是否缓存结果                      |
| `workers`     | 继承全局 | 最大并发数，`-1` 表示使用全部 CPU |
| `local`       | `False`  | 是否在主线程执行                  |
| `reduce_dims` | `()`     | 归约维度，`"all"` 表示全部        |
| `ignore`      | `None`   | 不计入缓存键的参数名列表          |

## @node.dimension()

定义维度节点。

| 参数   | 默认值 | 说明     |
| ------ | ------ | -------- |
| `name` | 函数名 | 维度名称 |

## Node

节点对象的主要方法。

| 方法                | 说明                       |
| ------------------- | -------------------------- |
| `node()`            | 执行 DAG 并返回结果        |
| `node(force=True)`  | 清除缓存后重新执行         |
| `node.invalidate()` | 清除该节点的缓存           |
| `repr(node)`        | 生成可复现的 Python 脚本   |
| `node.dims`         | 维度名元组（多维计算节点） |
| `node.coords`       | 坐标字典（多维计算节点）   |

## DimensionedResult

多维计算节点执行后返回的结果对象，继承自 `numpy.ndarray`。

| 属性/方法           | 说明                                     |
| ------------------- | ---------------------------------------- |
| `.dims`             | 维度名元组，与轴顺序对应                 |
| `.coords`           | 坐标字典 `{维度名: 坐标列表}`            |
| `.transpose(*dims)` | 按维度名转置，返回新的 DimensionedResult |
| `.items()`          | 迭代 `(元素, 坐标字典)` 对               |