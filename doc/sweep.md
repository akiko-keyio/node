# Sweep 设计与机制

`node.sweep` 是 Node 框架中用于进行参数扫描（Parameter Sweep）和超参数优化的核心工具。`sweep` 的核心设计理念是**通过配置驱动节点图的变异**。

## 核心概念

传统的 `map` 操作通常是在**函数参数**层面进行操作。例如，将 `process(x)` 应用于 `x=[1, 2, 3]`。这对于简单的数据处理是足够的。

然而，在涉及复杂的计算管线（Pipeline）时，许多参数并不直接作为函数参数传递，而是通过配置文件（Config）注入的，或者深埋在依赖链的上游节点中。

`sweep` 解决了这个问题：它不仅仅是改变函数的输入，而是**改变全局配置环境**，从而触发依赖于该配置的整个节点树（DAG）的重新实例化和计算。

## 机制解析

### 1. 配置注入与依赖

Node 框架的 `Config` 系统允许节点通过 `_target_` 和依赖注入（Dependency Injection）从配置中获取值或依赖的其他节点。

```python
# config.yaml (概念示意)
model:
  learning_rate: 0.01

trainer:
  _target_: my_module.Trainer
  lr: ${model.learning_rate}  # 引用上游配置
```

### 2. 迭代式扫描

当执行 `sweep` 时，框架会执行以下步骤：

1.  **准备配置列表**：接收用户提供的配置路径到值列表的映射（如 `{"model.learning_rate": [0.01, 0.05]}`）。
2.  **Zip 迭代**：将多个配置列表进行 `zip` 操作，而非笛卡尔积。这意味着所有提供的列表长度必须一致。第 `i` 次迭代将使用所有列表中的第 `i` 个元素。
3.  **环境切换与节点创建**：
    -   对于每次迭代，暂时修改全局配置 `node.cfg` 对应的值。
    -   **关键点**：在修改后的配置环境下，调用目标函数（Target Function）。
    -   由于配置变了，如果目标函数（或其依赖的节点）使用了 Config 注入的默认值，它们将解析到新的值。
    -   这将导致生成的 `Node` 对象具有不同的哈希值（Hash），从而被视为全新的节点。
4.  **结果收集**：收集所有生成的节点，并使用 `gather` 将它们打包成一个新的列表节点。

### 3. 生成脚本 (Script Generation)

由于每个扫描步骤都生成了独立的节点，`repr(node)` 生成的脚本将完整地反映出这一过程。所有配置的变化都被"烘焙"（Baked）到了生成的脚本中。

## 使用示例

### 基础扫描

```python
import node
from node import Config, sweep

# 1. 定义配置结构
config = Config({
    "global_param": 10,
    "process": {
        "threshold": "${global_param}" # 依赖全局参数
    }
})
node.configure(config=config)

# 2. 定义节点
@node.define()
def process(data, threshold):
    return [x for x in data if x > threshold]

# 3. 执行扫描
# 即使 process 函数签名中没有 'threshold' 参数传入，
# 但因为它在定义时绑定了 config 的默认值，
# 修改 "global_param" 会影响 "threshold"，进而改变 process 节点的哈希。
sweep_node = sweep(
    process,
    config={"global_param": [5, 10, 15]}, # 扫描值
    data=[1, 6, 11, 16] # 固定参数
)

results = sweep_node.get()
# 结果列表将包含三次执行的结果，分别对应 threshold=5, 10, 15
```

### 为什么不是笛卡尔积？

`sweep` 设计为 `zip` 迭代是为了提供最大的灵活性。如果需要笛卡尔积，用户可以在外部使用 `itertools.product` 生成参数列表，然后再传给 `sweep`。这也避免了框架在内部进行复杂的组合逻辑猜测。

## 局限性

-   **副作用**：`sweep` 修改的是全局配置状态。虽然通过单线程执行或适当的上下文管理（框架内部处理了这一部分）是安全的，但在多线程并发**构建**节点图时需要小心（执行阶段是安全的）。
-   **Config vs Args**：`sweep` 主要针对配置（Config）驱动的参数。如果要扫描的是显式的函数参数，可以使用 `gather` 配合列表推导来实现。

## 总结

`sweep` 是一个强大的工具，允许我们在**不改变代码**的情况下，通过**改变环境**来探索计算空间。这对于超参数搜索、敏感性分析和多场景模拟非常有用。
