# Node

为具有复杂依赖的计算任务设计的轻量级 Python DAG 计算框架。

## 适用任务

如果你的任务可以分解为一系列计算步骤，每个步骤由函数、参数和前序步骤的结果共同决定，Node 就能自动管理它们的缓存、执行顺序和配置。这些步骤在框架中称为**节点**，节点间的依赖关系构成 DAG（有向无环图）。

## 核心特性

- **增量计算** — 自动解析节点间依赖，仅在参数或上游依赖变化时重算并缓存。支持内存与磁盘多级缓存。
- **结果溯源** — 为每个缓存结果生成涵盖全部上游依赖与参数的可执行复现脚本。
- **配置管理** — 通过引用与组合动态构建层级配置，按节点名自动绑定，无需硬编码。
- **并行调度** — 自动分析依赖关系，将互不依赖的节点并行执行；可为单个节点设置最大并发数。
- **广播计算** — 为参数定义多组取值，框架自动对所有组合执行计算并独立缓存，取代手写循环。

## 安装

```bash
uv pip install -e .
```

Python ≥3.10。

## 快速开始

```python
import node

node.configure()

@node.define()
def add(x, y):
    return x + y

@node.define()
def square(n):
    return n * n

result = square(add(2, 3))  # 构建 DAG，不执行
print(result())             # 执行，输出 25
```

## 示例

**配置管理** — 参数写在 YAML 中，框架自动注入到同名函数：

```yaml
# config.yaml
load:
  path: "data.csv"
clean:
  threshold: 0.5
```

```python
from node import Config

node.configure(config=Config("config.yaml"), workers=4)

@node.define()
def load(path):
    return pd.read_csv(path)

@node.define()
def clean(data, threshold):
    return data[data.value > threshold]

@node.define()
def train(data, method):
    return fit_model(method, data)

@node.define()
def compare(a, b):
    return {"RF": score(a), "LR": score(b)}
```

**构建与执行** — 调用节点函数只构建 DAG，`()` 才真正执行：

```python
cleaned = clean(data=load())              # load、clean 的参数从配置注入
report = compare(
    a=train(data=cleaned, method="RF"),
    b=train(data=cleaned, method="LR"),
)
report()  # 执行整条管线
```

**自动缓存** — 再次执行或只改部分参数时，未变化的节点直接返回缓存：

```python
report()  # 全部命中缓存，瞬间返回

# 只修改 threshold，重新构建管线
node.cfg.clean.threshold = 0.8
report = compare(
    a=train(data=clean(data=load()), method="RF"),
    b=train(data=clean(data=load()), method="LR"),
)
report()  # load 命中缓存；clean、train、compare 因上游变化而重算
```

**并行调度** — `train("RF")` 和 `train("LR")` 互不依赖，自动并行执行。

**结果溯源** — 生成涵盖完整依赖链的可执行脚本：

```python
print(repr(report))
# load_0 = load(path="data.csv")
# clean_0 = clean(data=load_0, threshold=0.8)
# train_0 = train(data=clean_0, method="RF")
# train_1 = train(data=clean_0, method="LR")
# compare_0 = compare(a=train_0, b=train_1)
```

**广播计算** — 用维度替代手写循环，自动展开所有取值并独立缓存：

```python
@node.dimension()
def method():
    return ["RF", "LR", "SVM"]

result = train(data=cleaned, method=method())()
# 自动执行 3 次 train，结果为 DimensionedResult
# result.dims = ("method",)，可按维度索引和聚合
```

## 文档

- [用户指南](docs/user-guide.md) — 完整使用手册
- [API 参考](docs/api.md) — 接口与参数速查
