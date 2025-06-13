# Node

Node 是一个轻量级的 DAG 流程库，提供内存和磁盘两级缓存以加速计算。

## 安装

建议在虚拟环境中安装。克隆本仓库后执行：

```bash
pip install -e .
```

## 快速开始

创建 `Flow`，将普通函数包装成任务并构建 DAG，最后运行：

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

`DiskJoblib` 会以函数名创建子目录，优先将结果保存为 `<expr>.pkl`，其中
`<expr>` 为节点表达式经简单清理后的字符串。当文件名过长或包含非法字符时，
会退回保存为 `<hash>.pkl`，并在同目录写入 `<hash>.py` 记录 `repr(node)` 以便查
看。例如运行 `tutorial.py` 后可能会看到：

```
.cache/inc/inc_3_.pkl
.cache/add/7d04d898305ce53d9c744df7d005317b.pkl
.cache/add/7d04d898305ce53d9c744df7d005317b.py
```

第一个文件使用表达式命名，第二个文件因为名称不合法或过长而使用哈希值。


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

## 教程脚本

阅读并运行 `tutorial.py` 可以查看完整的示例，包括：

1. 构建和运行简单的 DAG；
2. 从 YAML 中加载任务默认参数；
3. 在磁盘上查看缓存文件的命名方式以及生成的 `.py` 脚本。

执行：

```bash
python tutorial.py
```

日志会展示每个任务的执行及缓存情况。

## 运行测试

确保你已安装完所需依赖，在项目根目录下运行:

```bash
pytest
```
13 个测试全部通过即表示环境配置正确。
