# Node

Node 是一个轻量级的 DAG 流程库，提供内存和磁盘两级缓存以加速计算。

## 安装

先克隆本仓库，然后在下面命令组中执行：

```bash
pip install -e .
```

## 快速开始

创建 `Flow`，将普通函数包装成任务并构建 DAG 后执行：

```python
from node.node import Flow

flow = Flow()

@flow.task()
def add(x, y):
    return x + y

@flow.task()
def square(z):
    return z * z

result = flow.run(square(add(2, 3)))
print(result)  # 25
```

## 查看脚本

每个节点对象在 `repr` 时都会生成等效的执行脚本，便于调试和分享：

```python
root = square(add(2, 3))
print(repr(root))
# n0 = add(2, 3)
# n1 = square(n0)
# n1
```

## 缓存与并行

Flow 默认使用 `MemoryLRU`和 `DiskJoblib`存储结果。你可以给定 `ChainCache`，也可通过 `executor` 和 `workers` 控制并行计算量。

```python
from node.node import Flow, ChainCache, MemoryLRU, DiskJoblib

flow = Flow(
    cache=ChainCache([MemoryLRU(), DiskJoblib(".cache")]),
    executor="thread",
    workers=4,
)
```

## 配置对象

可以给 `Flow` 传入 `Config` 实例，以集中管理任务的默认参数。配置通常来自一个 YAML 文件：

```python
import yaml
from node.node import Flow, Config

with open("defaults.yml") as f:
    defaults = yaml.safe_load(f)

flow = Flow(config=Config(defaults))
```

```yaml
# defaults.yml
add:
  y: 5
```

项目的测试目录中也包含 `tests/config.yaml` 作为参考。

## 运行测试

确保你已安装完所需依赖，在项目根目录下运行:

```bash
pytest
```
13 个测试全部通过即表示环境配置正确。
