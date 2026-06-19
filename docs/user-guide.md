# Node 用户指南

本文档是 Node 框架的完整使用手册。接口详情见 [API 参考](api.md)。

---

## 1. 入门

### 1.1 安装

```bash
uv pip install -e .
```

需要 Python ≥3.10。

### 1.2 初始化

使用前必须调用 `node.configure()`，且整个进程内只能调用一次。需要重新配置时先调用 `node.reset()`。

```python
import node

node.configure()  # 使用默认缓存与线程执行器
```

常用初始化参数：

```python
from node import Config

node.configure(
    config=Config("config.yaml"),  # 可选：加载 YAML 配置
    workers=4,                       # 默认并发数
)
```

### 1.3 定义与执行节点

用 `@node.define()` 将普通函数注册为节点工厂。调用节点函数时只构建 DAG，不执行计算；再调用 `()` 才会真正运行：

```python
@node.define()
def add(x, y):
    return x + y

@node.define()
def square(n):
    return n * n

result = square(add(2, 3))  # 构建 DAG
print(result())             # 执行并输出 25
```

**计算节点**由三部分决定：函数（计算逻辑）、参数（影响计算的值）、依赖（上游节点的结果）。三者相同即为同一计算。多个节点按依赖关系组成 DAG（有向无环图）。

---

## 2. 自动缓存

框架自动缓存计算结果。第二次执行时，若输入不变则直接返回缓存：

```python
@node.define()
def slow_compute(x):
    time.sleep(2)
    return x * 2

slow_compute(5)()  # 首次执行：约 2 秒
slow_compute(5)()  # 缓存命中：瞬间返回
slow_compute(6)()  # 参数变化：重新计算
```

### 2.1 缓存键

每个节点的缓存键由以下因素决定：

```
缓存键 = hash(函数名, sorted((参数名, 参数值), ...))
```

依赖节点类型的参数值使用该节点的缓存键表示；普通参数使用确定性序列化。参数按名称排序，保证跨运行一致。

**函数名**标识计算逻辑；**函数体变更不会自动使缓存失效**。修改实现后需手动清理：

```python
result.invalidate()                    # 仅清除该节点
result.invalidate(recursive=True)        # 递归清除依赖链
```

### 2.2 纯函数要求

节点函数应为纯函数（相同输入 → 相同输出），否则缓存可能不正确：

```python
# ✓ 纯函数
@node.define()
def process(data, threshold):
    return [x for x in data if x > threshold]

# ✗ 闭包变量不在缓存键中
def make_processor(factor):
    @node.define()
    def process(x):
        return x * factor  # factor 变化不会触发重算
    return process

# ✗ 非确定性输出
@node.define()
def random_data(n):
    return [random.random() for _ in range(n)]
```

### 2.3 禁用缓存

```python
@node.define(cache=False)
def get_current_time():
    return datetime.now()
```

### 2.4 排除参数

不影响结果的参数（如 `debug`、`verbose`）应使用 `ignore` 排除在缓存键之外：

```python
@node.define(ignore=["debug", "verbose"])
def compute(x, debug=False, verbose=False):
    if debug:
        print(f"Computing {x}")
    return x * 2
```

### 2.5 缓存后端

默认使用两级缓存：内存 LRU → 磁盘。

```python
from node import ChainCache, MemoryLRU, DiskCache

node.configure(cache=ChainCache([
    MemoryLRU(maxsize=512),
    DiskCache(root=".cache"),
]))
```

| 类型 | 用途 | 配置 |
|------|------|------|
| `MemoryLRU` | 热数据快速访问 | `maxsize`：最大条目数 |
| `DiskCache` | 冷数据持久化 | `root`：缓存目录 |
| `ChainCache` | 多级组合 | 按顺序查找，命中后回填上级 |

### 2.6 缓存清理

不要暴力删除整个 `.cache/` 目录。推荐局部清理：

```bash
rm -rf .cache/my_function_name/
```

框架不检测函数代码变更；修改逻辑后需 `invalidate()` 或清理对应目录。

广播产生的子节点与父节点共用同一 `cache` 语义：`cache=True/False` 同时作用于当前节点及其广播子节点。

---

## 3. 结果溯源

缓存键是哈希值，不可逆。为支持调试复现，可对任意节点调用 `repr()` 获得可执行的 Python 脚本：

```python
print(repr(square(add(2, 3))))
```

输出示例：

```python
# hash = 7c3b8fad541e11
add_0 = add(x=2, y=3)
square_0 = square(n=add_0)
```

使用 `DiskCache` 时，脚本会自动保存在缓存目录：

```
.cache/
├── square/
│   ├── 7c3b8fad541e11.pkl    # 计算结果
│   └── 7c3b8fad541e11.py     # 复现脚本
```

可将 `repr()` 输出保存为独立脚本，在本地复现问题，无需依赖完整框架运行时。

---

## 4. 配置管理

### 4.1 加载配置

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

### 4.2 参数自动注入

配置中与函数同名的节会自动注入为该函数的默认参数：

```python
@node.define()
def load_data(path, threshold):
    df = pd.read_csv(path)
    return df[df.value > threshold]

load_data()()                    # 使用配置值
load_data(path="other.csv")()    # 调用时覆盖
```

### 4.3 运行时修改

```python
node.cfg.load_data.path = "custom.csv"
load_data()()
```

### 4.4 instantiate()

`node.instantiate("xxx")` 在调用时读取当前配置并构建已绑定参数的节点。之后修改 `node.cfg` 不会自动更新已返回的节点；需重新 `instantiate()`。

临时改参而不污染全局配置，使用 `overrides`：

```python
grid = node.instantiate(
    "train",
    overrides={
        "train.optimizer": "sgd",
        "train.lr": 0.001,
    },
)
result = grid()
```

### 4.5 变量引用

```yaml
base_dir: "/data"
raw_path: "${base_dir}/raw.csv"
processed_path: "${base_dir}/out.csv"
```

### 4.6 节点依赖

配置节含 `_target_` 时，框架按路径构建节点并注入依赖：

```yaml
load_data:
  _target_: mymodule.load_data
  path: "data.csv"

process:
  _target_: mymodule.process
  data: ${load_data}
  threshold: 0.5
```

---

## 5. 并行调度

执行一个节点时，框架自动分析其整棵依赖树，将互不依赖的上游节点提交到线程池并行执行：

```python
node.configure(workers=4)

@node.define()
def download(url):
    return requests.get(url).text

@node.define()
def merge(a, b, c):
    return combine(a, b, c)

# download 三个节点互不依赖，merge 执行时它们并行运行
merge(download(url1), download(url2), download(url3))()
```

当前仅支持 `"thread"` 执行器，适用于 I/O 密集型任务。CPU 密集型任务可配合 `limit_inner_parallelism=True`（默认）限制节点内部 BLAS/OpenMP 线程，防止线程爆炸。

### 5.1 节点级并发控制

| 配置 | 行为 |
|------|------|
| 不指定 | 使用 `configure(workers=N)` 的全局设置 |
| `workers=2` | 该函数最多 2 个实例同时运行 |
| `workers=-1` | 使用全部 CPU 核心 |
| `local=True` | 在主线程执行，不进入线程池 |

```python
@node.define(workers=2)
def heavy_task(x):
    return expensive_computation(x)

@node.define(local=True)
def gui_task():
    update_ui()
```

菱形依赖中，共享上游节点只执行一次：

```
    A
   / \
  B   C    → A 执行一次，B/C 并行，D 等待 B/C
   \ /
    D
```

### 5.2 错误处理

默认 `continue_on_error=False`：任一节点失败时立即终止，便于调试。

```python
node.configure(continue_on_error=True)

@node.define()
def may_fail(x):
    if x < 0:
        raise ValueError("x must be positive")
    return x

a = may_fail(-1)  # 失败
b = may_fail(1)   # 仍执行
result = downstream(a, b)  # 因依赖 a 被跳过
```

---

## 6. 广播计算

对参数的多个取值执行相同计算时，无需手写循环。定义**维度**节点表示一组取值，传入普通节点后框架自动展开计算。

### 6.1 定义维度

```python
@node.dimension()
def time():
    return [2020, 2021, 2022]
```

维度名默认为函数名；坐标为函数返回的列表。

### 6.2 广播与对齐

**广播**（不同维度）：自动计算笛卡尔积。

```python
@node.dimension()
def model():
    return ["LR", "RF"]

@node.define()
def train(t, m):
    return fit(m, t)

grid = train(t=time(), m=model())  # 3×2 = 6 个独立计算
```

**对齐**（相同维度）：自动 zip，而非笛卡尔积。

```python
t = time()
s1 = step1(t=t)
s2 = step2(x=s1, t=t)  # 与 step1 共享 time 维度
```

### 6.3 结果访问

执行后返回 `DimensionedResult`（`numpy.ndarray` 子类），携带维度元数据：

```python
result = load(t=time())()

result.dims    # ("time",)
result.coords  # {"time": [2020, 2021, 2022]}
result.shape   # (3,)

result[0]      # 按索引访问
grid.transpose("time", "model")
```

按坐标遍历：

```python
import numpy as np

for idx in np.ndindex(result.shape):
    item = result[idx]
    coord = {d: result.coords[d][idx[i]] for i, d in enumerate(result.dims)}
```

### 6.4 聚合

使用 `reduce_dims` 归约指定维度：

```python
@node.define(reduce_dims="all")
def summary(data):
    return {"count": len(data.flat), "total": sum(d["value"] for d in data.flat)}

@node.define(reduce_dims=["time"])
def site_avg(data):
    return sum(d["value"] for d in data.flat) / len(data.flat)
```

`reduce_dims` 节点收到的 `data` 始终是 `DimensionedResult`；声明 `reduce_dims=["time", "model"]` 时，`data.dims` 严格等于 `("time", "model")`。

### 6.5 sweep 参数扫描

临时实验可用 `instantiate(..., sweep=...)` 做笛卡尔积扫描，无需定义维度节点：

```python
grid = node.instantiate(
    "train",
    sweep={
        "train.optimizer": ["adam", "sgd"],
        "train.lr": [0.01, 0.001, 0.0001],
    },
)
result = grid()  # shape=(2, 3), dims=("sweep_optimizer", "sweep_lr")
```

`sweep` 的 key 须为全局配置路径（`section.param`）。长期存在的业务维度建议用 `@node.dimension()`。

---

## 7. 最佳实践

### 7.1 推荐项目结构

```text
my_project/
├── config.yaml
├── pyproject.toml
├── src/my_module/
│   ├── __init__.py      # node.configure(config=...)
│   ├── nodes.py         # 稳定节点定义
│   └── utils.py
├── exps/01_baseline/    # 实验目录（自包含）
└── tests/
```

在包 `__init__.py` 中初始化：

```python
from pathlib import Path
import node
from node import Config

PROJ_ROOT = Path(__file__).parents[2]
node.configure(config=Config(PROJ_ROOT / "config.yaml"))
```

### 7.2 核心理念

- **配置驱动**：参数与超参数写入 YAML，代码只定义逻辑。
- **显式输入**：影响计算的量必须通过函数参数传入，勿在节点内读全局配置或环境变量。
- **细粒度拆分**：小节点提高缓存命中率、并行度和可调试性。
- **类型注解**：为参数与返回值添加类型，便于静态分析与 IDE。

### 7.3 常见陷阱

- **闭包**：避免在函数内定义节点；闭包捕获的变量不计入缓存键。
- **不可变参数**：优先使用 tuple、str、frozenset；复杂自定义对象可能导致哈希不稳定。
- **循环依赖**：依赖必须构成 DAG，不可 A→B→A。

---

## 8. API 参考

完整的接口文档见 [API 参考](api.md)。

---

## 9. 完整示例

```python
import node
import numpy as np

node.configure()

@node.dimension()
def time():
    return [2020, 2021, 2022]

@node.dimension()
def model():
    return ["LR", "RF"]

@node.define()
def load(t):
    np.random.seed(t)
    return np.random.randn(100)

@node.define()
def train(data, m):
    score = np.mean(data) + (0.1 if m == "RF" else 0)
    return {"model": m, "score": score}

@node.define(reduce_dims="all")
def report(data):
    scores = [d["score"] for d in data.flat]
    return {
        "best_score": max(scores),
        "avg_score": sum(scores) / len(scores),
        "model_count": len(scores),
    }

times = time()
models = model()
result = report(data=train(data=load(t=times), m=models))()
print(result)

trained = train(data=load(t=times), m=models)()
print(f"训练结果维度: {trained.dims}")  # ("model", "time")

for idx in np.ndindex(trained.shape):
    item = trained[idx]
    coord = {d: trained.coords[d][idx[i]] for i, d in enumerate(trained.dims)}
    print(f"  {coord['model']}-{coord['time']}: {item['score']:.3f}")
```
