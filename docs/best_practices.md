# Node 框架最佳实践

本文档基于 `Node` 框架的设计理念和源码分析，总结了构建高质量数据管线和计算任务的最佳实践。遵循这些准则将有助于创建可维护、高性能且易于调试的系统。

## 1. 核心理念

Node 框架的核心在于 **"配置驱动"** 和 **"纯函数计算"**。

*   **配置驱动 (Config-Driven)**: 所有的参数、超参数和依赖关系应定义在 YAML 配置文件中，而不是硬编码在 Python 代码里。代码只负责定义"怎么做" (Logic)，配置负责定义"做什么" (Data & Params)。
*   **声明式编程 (Declarative)**: 用户只需定义计算节点 (Node) 及其依赖，框架自动处理执行顺序、并行调度和缓存。

## 2. 项目结构

推荐采用以下项目结构，以保持代码组织清晰：

```text
my_project/
├── config.yaml           # 主配置文件
├── pyproject.toml        # 依赖管理
├── src/
│   └── my_module/
│       ├── __init__.py   # 包入口，在此处调用 node.configure()
│       ├── nodes.py      # 核心/稳定节点定义 (Production Ready)
│       └── utils.py      # 纯辅助函数
├── exps/                 # 实验目录 (Experiments)
│   ├── 01_baseline/      # 实验 ID/Name
│   │   ├── run_exp.py    # 实验运行脚本
│   │   ├── exp_nodes.py  # 实验专用节点
│   │   ├── analysis.py   # 分析脚本
│   │   ├── report.md     # 实验报告
│   │   └── figures/      # 输出图片 (报告中引用: ![fig](figures/plot.png))
│   └── ...
├── tests/                # 测试代码
└── run.py                # 入口脚本
```

### 2.1 框架初始化
建议在包的 `__init__.py` 中定义全局 `PROJ_ROOT` 并进行框架初始化。

```python
# src/my_module/__init__.py
from pathlib import Path
import node
from node import Config

# 定义项目根目录 (假设 __init__.py 在 src/my_module/ 下)
PROJ_ROOT = Path(__file__).parents[2]

# 初始化 Node 框架
# 直接使用 PROJ_ROOT 定位配置文件，后续代码中涉及路径也应基于 PROJ_ROOT
node.configure(config=Config(PROJ_ROOT / "config.yaml"))
```

### 2.2 实验管理 (Experiments)
对于探索性工作，建议在 `exps/` 目录下为每个实验创建独立文件夹。

**结构建议**:
*   **自包含**: 每个实验文件夹包含该实验所需的脚本、专用节点定义、分析代码和结果。
*   **引用稳定组件**: 在实验脚本中导入 `src.my_module` 中的稳定节点，复用已有能力。
*   **报告闭环**: 实验产出的图片直接保存在该文件夹下（如 `figures/`），并在同级 `report.md` 中引用 (`![title](figures/plot.png)`), 形成完整的实验记录。

**运行建议**:
统一使用 `uv` 包管理器运行脚本，它会自动处理环境和路径问题：

```bash
# 在项目根目录下使用 uv 运行
$ uv run exps/01_baseline/run_exp.py
```

## 3. 节点定义 (Node Definition)

### 3.1 保持函数“纯粹” (Pure Functions)
节点函数应尽量为 **纯函数**：
*   **输出仅依赖于输入**: 相同的输入参数必然产生相同的输出。
*   **无副作用**: 不要在节点内修改全局变量或外部状态（除必要的 I/O 操作外，如写文件，但应尽量让框架管理 I/O）。
*   **优势**: 确保缓存机制正确工作，避免"缓存失效"导致的诡异 Bug。

### 3.2 显式输入 (Explicit Inputs)
所有影响计算的变量都必须作为参数传入函数。
*   **Do**:
    ```python
    @node.define()
    def process(data, threshold): ...
    ```
*   **Don't**: 在函数内部读取全局配置 `CONF.threshold` 或环境变量。这会导致框架无法感知依赖变化，从而无法正确更新缓存。

### 3.3 细粒度 (Granularity)
将大任务拆分为小的、可复用的节点。
*   **优势**:
    *   **缓存命中率高**: 小步骤的变化只触发该步骤及下游重算，上游大计算量步骤可直接复用缓存。
    *   **并行度高**: 更多的小任务意味着更多的并行机会。
    *   **易于调试**: 中间结果可独立检查。

### 3.4 类型提示 (Type Hinting)
始终为节点参数和返回值添加类型注解。虽然框架运行时不强制，但这有助于静态分析和 IDE 提示。

```python
@node.define()
def calculate_metrics(data: pd.DataFrame, method: str) -> dict[str, float]:
    ...
```

## 4. 配置管理 (Configuration)

### 4.1 善用 `_target_` 进行依赖注入
在 YAML 中使用 `_target_` 指定函数，配合 `${node}` 语法自动管理依赖。这使得管线结构在配置中一目了然。

```yaml
# config.yaml
load_data:
  _target_: my_module.nodes.load_csv
  path: "data/input.csv"

process_data:
  _target_: my_module.nodes.clean
  data: ${load_data}  # 自动注入 load_data 节点的输出
  threshold: 0.5
```

### 4.2 使用预设 (Presets) 管理环境
利用 `_presets_` 和 `_use_` 机制在开发 (Dev) 和生产 (Prod) 环境间无缝切换。

```yaml
train_model:
  _target_: my_module.nodes.train
  _use_: dev
  _presets_:
    dev:
      epochs: 10
      batch_size: 32
    prod:
      epochs: 1000
      batch_size: 256
```

运行时可通过 `node.cfg.train_model._use_ = "prod"` 动态切换，无需修改代码。

## 5. 性能与缓存 (Performance & Caching)

### 5.1 使用 `ignore` 排除非功能参数
对于不影响计算结果的参数（如 `verbose`, `debug`, `n_jobs` 等），应使用 `ignore` 排除在缓存键计算之外。

```python
@node.define(ignore=["verbose", "workers"])
def heavy_compute(data, verbose=False, workers=4):
    ...
```
否则，仅仅修改 `verbose=True` 就会导致昂贵的计算任务重跑。

### 5.2 合理选择执行器
*   **I/O 密集型**: 使用 `executor="thread"` (默认)。适合网络请求、文件读写。
*   **CPU 密集型**: 使用 `executor="process"`。因为 Python GIL 的存在，CPU 密集型任务必须用多进程才能并行。

### 5.3 缓存清理
不要暴力删除 `.cache/` 目录。
*   **局部清理**: 使用 `rm -rf .cache/my_function_name/` 清理特定节点的缓存。
*   **代码变更**: 框架目前不会检测函数代码变更。如果修改了逻辑，必须手动 `invalidate()` 或清理相关缓存目录。

## 6. 多维计算 (Multi-dimensional Computation)

Node 框架的杀手级特性是内置的**多维计算**支持。它允许你通过定义“维度”来自动处理参数遍历，而不是在代码外部写 `for` 循环。

### 6.1 定义维度 (`@node.dimension`)
维度是特殊的节点，用于定义一个参数的一组取值（坐标）。

```python
# 定义 "time" 维度，包含 3 个坐标
@node.dimension()
def time():
    return [2020, 2021, 2022]

# 定义 "model" 维度，包含 2 个坐标
@node.dimension()
def model():
    return ["A", "B"]
```

### 6.2 自动广播 (Broadcasting)
当你将维度节点作为参数传递给普通节点时，框架会自动执行广播（笛卡尔积）。

*   **Bad (外部循环)**:
    ```python
    results = []
    for t in [2020, 2021, 2022]:
        for m in ["A", "B"]:
            # 手动循环生成 6 个节点，繁琐且难以管理结果
            results.append(train(time=t, model=m))
    ```

*   **Good (维度广播)**:
    ```python
    # 一行代码，自动生成 6 个并行计算任务
    # 返回结果自动封装为 DimensionedResult
    grid = train(time=time(), model=model()) 
    ```

### 6.3 结果处理
多维计算返回的并不是普通列表，而是 `DimensionedResult` (类 Numpy 数组)，保留了维度元数据。

```python
result = grid() # 执行计算

# 像 numpy 数组一样切片
print(result.shape)      # (3, 2)
print(result.dims)       # ('time', 'model')
print(result.coords)     # {'time': [2020...], 'model': ['A', 'B']}

# 获取特定坐标结果
data_2020_A = result[0, 0]
```

使用多维计算，可以让你的代码摆脱繁琐的循环控制逻辑，专注于通过维度组合来表达计算意图。

### 6.4 聚合结果 (Aggregation)
多维计算产生的结果往往需要汇总（如计算平均值、选出最优解）。使用 `reduce_dims` 参数可轻松实现。

```python
# 自动汇聚所有模型的结果，计算平均分数
@node.define(reduce_dims="model")
def average_score(results):
    # results 是一个列表 (因为 model 维度被归约了)
    return sum(r["score"] for r in results) / len(results)

grid = train(time=time(), model=model())
final_metric = average_score(grid) # 此时 final_metric 只保留 time 维度
```


## 7. 调试与复现 (Debugging)

Node 框架提供了强大的复现能力。对于任何节点 `n`，调用 `repr(n)` 都会返回一段**完全独立的可执行脚本**。

### 7.1 生成复现脚本
当实验在服务器上失败时，无需重新运行整个流程。

```python
n = some_complex_calculation(...)
print(repr(n))
```

输出示例：
```python
# hash = 7c3b8fad...
load_data_0 = load_data(path="data.csv")
process_0 = process(data=load_data_0, threshold=0.5)
# ... 依赖链完全展开 ...
```

**最佳实践**: 将这段输出保存为 `debug_script.py`，在本地直接运行即可复现问题。这使得调试与环境解耦，极大地提升了排查效率。

## 8. 常见陷阱 (Pitfalls)

*   **闭包陷阱**: 避免在函数内定义节点函数（闭包）。因为闭包捕获的外部变量不会被计入缓存键，导致外部变量变化时缓存未失效。
*   **不可变参数**: 尽量使用不可变对象（tuple, str, frozenset）作为参数。如果必须使用 dict 或 list，框架会自动处理 canonicalization (规范化)，但复杂的自定义对象可能导致哈希不稳定，从而导致缓存频繁失效。
*   **循环依赖**: Node 依赖必须是有向无环图 (DAG)。避免 A 依赖 B，B 又依赖 A。

---
遵循以上实践，您将能够充分利用 Node 框架的强大功能，构建出稳定、高效的数据处理系统。
