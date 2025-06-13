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

`DiskJoblib` 会将结果保存为 `<hash>.pkl`，并在同目录生成相同名字的 `.py` 文件，
其中包含节点对象的 `repr`，方便查看缓存内容。例如运行 `tutorial.py` 后可能会
看到如下文件：

```
.cache/fa/fa817707222760b0a0c23433a19cfdb8.pkl
.cache/fa/fa817707222760b0a0c23433a19cfdb8.py
```

这里目录 `fa` 是哈希值的前两位，文件名是对应节点签名的 MD5 值。

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
