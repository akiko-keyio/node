# API 参考

## node.configure()

初始化全局运行时。整个进程内只能调用一次；需要重新配置时先调用 `node.reset()`。

```python
node.configure(
    config=None,
    cache=None,
    executor="thread",
    workers=4,
    continue_on_error=False,
    validate=True,
    limit_inner_parallelism=True,
)
```

**参数**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `config` | `Config \| str \| None` | `None` | `Config` 对象或 YAML 文件路径 |
| `cache` | `Cache \| None` | `ChainCache([MemoryLRU(), DiskCache()])` | 缓存后端实例 |
| `executor` | `Literal["thread"]` | `"thread"` | 执行器类型，当前仅支持 `"thread"` |
| `workers` | `int` | `4` | 默认并发线程数 |
| `continue_on_error` | `bool` | `False` | `True` 时单节点失败跳过下游继续执行；`False` 时立即终止 |
| `validate` | `bool` | `True` | 是否使用 Pydantic 校验函数参数类型 |
| `limit_inner_parallelism` | `bool` | `True` | 限制节点内部 BLAS/OpenMP 线程数，防止线程爆炸 |

**返回值**

`Runtime` — 配置完成的运行时实例。

---

## node.reset()

重置全局运行时，允许重新调用 `configure()`。主要用于测试。

```python
node.reset()
```

---

## @node.define()

将普通函数转换为节点工厂。装饰后调用函数返回 `Node` 对象（构建 DAG），不立即执行。

```python
@node.define(
    cache=True,
    workers=None,
    local=False,
    reduce_dims=(),
    ignore=None,
)
def my_func(...):
    ...
```

**参数**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `cache` | `bool` | `True` | 是否缓存该节点及其广播子节点的结果 |
| `workers` | `int \| None` | `None`（继承全局） | 最大并发实例数；`-1` 使用全部 CPU 核心 |
| `local` | `bool` | `False` | `True` 时在主线程执行，不提交到线程池 |
| `reduce_dims` | `Sequence[str] \| str` | `()` | 归约维度名；`"all"` 归约所有维度 |
| `ignore` | `list[str] \| None` | `None` | 不计入缓存键的参数名列表 |

**注意事项**

- 框架会对使用闭包的函数发出警告，因为闭包变量不计入缓存键。
- 启用 `validate=True`（全局配置）时，函数参数会经过 Pydantic 校验。

---

## @node.dimension()

将函数定义为维度节点。函数返回一个列表，表示该维度的所有坐标值。

```python
@node.dimension(name=None)
def my_dim():
    return [...]
```

**参数**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | `str \| None` | `None`（使用函数名） | 维度名称 |

**返回值**

装饰后的函数调用时返回 `Node`，携带 `dims` 和 `coords` 属性。

---

## node.instantiate()

从当前配置中按 section 名构建已绑定参数的节点。调用时读取配置快照，后续 `node.cfg` 修改不影响已返回的节点。

```python
node.instantiate(
    name,
    overrides=None,
    sweep=None,
)
```

**参数**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | `str` | — | 配置 section 名称 |
| `overrides` | `Mapping[str, Any] \| None` | `None` | 一次性参数覆盖，不修改全局配置。Key 格式为 `"section.param"` |
| `sweep` | `Mapping[str, Any] \| None` | `None` | 参数扫描，返回笛卡尔积维度节点。Key 格式为 `"section.param"` |

**返回值**

`Node` — 当指定 `sweep` 时返回带维度信息的节点。

---

## node.cfg

全局配置代理对象，支持属性式读写。

```python
node.cfg.load_data.path           # 读取
node.cfg.load_data.path = "new"   # 修改
```

修改会影响后续 `instantiate()` 调用，但不影响已创建的节点。

---

## Config

配置类，加载并管理 YAML 配置。

```python
from node import Config

config = Config("config.yaml")
config = Config({"section": {"key": "value"}})
```

**构造参数**

| 参数 | 类型 | 说明 |
|------|------|------|
| `mapping` | `Mapping \| DictConfig \| str \| Path \| None` | YAML 文件路径、字典或 OmegaConf 对象 |

---

## Node

计算节点，表示 DAG 中的一个计算单元。通常由 `@node.define()` 装饰的函数调用时自动创建，不需要手动实例化。

### 执行

| 用法 | 说明 |
|------|------|
| `node()` | 执行 DAG 并返回结果 |
| `node(force=True)` | 清除相关缓存后重新执行 |

### 缓存管理

| 用法 | 说明 |
|------|------|
| `node.invalidate()` | 清除该节点的缓存 |
| `node.invalidate(recursive=True)` | 递归清除该节点及所有上游依赖的缓存 |

### 溯源

| 用法 | 说明 |
|------|------|
| `repr(node)` | 返回可复现的 Python 脚本字符串 |
| `node.script` | 与 `repr()` 相同，返回脚本字符串 |
| `node.script_lines` | 返回 `[(hash, line), ...]` 列表 |

### 维度属性（广播节点）

| 属性 | 类型 | 说明 |
|------|------|------|
| `node.dims` | `tuple[str, ...]` | 维度名元组，与数组轴顺序对应 |
| `node.coords` | `dict[str, list]` | 坐标字典 `{维度名: 取值列表}` |

---

## DimensionedResult

广播计算的执行结果，继承自 `numpy.ndarray`，额外携带维度元数据。

### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `.dims` | `tuple[str, ...]` | 维度名元组，与数组轴顺序对应 |
| `.coords` | `dict[str, list]` | `{维度名: 坐标列表}` |
| `.shape` | `tuple[int, ...]` | 数组形状，轴顺序与 `.dims` 一致 |
| `.flat` | `iterator` | 扁平迭代器，遍历所有元素 |

### 方法

| 方法 | 说明 |
|------|------|
| `.transpose(*dims)` | 按维度名重排轴顺序，返回新的 `DimensionedResult` |

**示例**

```python
result = train(t=time(), m=model())()

result.dims       # ("model", "time")
result.shape      # (2, 3)
result.coords     # {"model": ["LR", "RF"], "time": [2020, 2021, 2022]}

result[0, 1]      # model=LR, time=2021
result[1, :]      # model=RF 的全部时间
result.transpose("time", "model")   # 转置为 (3, 2)
```

---

## 缓存后端

### MemoryLRU

内存 LRU 缓存。

```python
from node import MemoryLRU

cache = MemoryLRU(maxsize=512)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `maxsize` | `int` | `512` | 最大缓存条目数 |

### DiskCache

磁盘缓存，使用 pickle 序列化。自动保存复现脚本（`.py`）。

```python
from node import DiskCache

cache = DiskCache(root=".cache", lock=True)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `root` | `str \| Path` | `".cache"` | 缓存目录路径 |
| `lock` | `bool` | `True` | 是否使用文件锁防止并发写入冲突 |

缓存目录结构：

```
.cache/
└── <函数名>/
    ├── <hash>.pkl    # pickle 序列化的计算结果
    └── <hash>.py     # 可复现的计算脚本
```

### ChainCache

多级缓存链，按顺序查找，命中后回填上级缓存。

```python
from node import ChainCache, MemoryLRU, DiskCache

cache = ChainCache([
    MemoryLRU(maxsize=512),
    DiskCache(root=".cache"),
])
```

---

## 异常

所有异常继承自 `NodeError`。

| 异常类 | 说明 |
|--------|------|
| `NodeError` | 框架异常基类 |
| `ConfigurationError` | 配置解析或结构错误 |
| `DimensionMismatchError` | 维度不兼容（广播或归约时） |
| `CacheError` | 缓存操作失败 |

```python
from node import NodeError, ConfigurationError, DimensionMismatchError, CacheError
```
