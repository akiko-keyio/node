# Description

当使用 Node 库（轻量级 Python DAG 执行框架）构建数据管线或计算任务时参考此规则。

# Content

## ⚠️ 核心原则

### 配置驱动

1. **所有参数写入 YAML**：节点默认参数全部定义在配置文件中，代码不硬编码参数值
2. **显式输入**：影响计算结果的参数必须通过函数签名显式声明，不使用全局变量或闭包
3. **直接调用执行**：需要结果时直接 `node_name()()` 即可，框架自动处理依赖、缓存、并行

```python
# ✓ 推荐：配置驱动
node.configure(config="config.yaml")
result = process()()  # 参数从配置注入，直接调用

# ✗ 避免：硬编码参数
result = process(path="data.csv", threshold=0.5)()
```

### 缓存清除警告

缓存目录结构：`.cache/<函数名>/<hash>.pkl`

修改节点逻辑后如需清除缓存：

```bash
# ✓ 只清除特定节点的缓存
rm -rf .cache/process/

# ✗ 绝对不要清空整个缓存目录！
rm -rf .cache/  # 危险！其他节点缓存重建代价巨大
```

---

## 缓存

框架自动缓存计算结果：

```python
slow_compute(5)()  # 首次执行：2秒
slow_compute(5)()  # 缓存命中：瞬间返回
slow_compute(6)()  # 参数变化：重新计算
```

**缓存键计算**

```
缓存键 = hash(函数名 + 参数值 + 依赖节点的缓存键)
```

- 函数名标识计算逻辑，**函数体变更不会自动使缓存失效**
- 需手动 `result.invalidate()` 清除缓存

**纯函数要求**

```python
# ✓ 纯函数：相同输入 → 相同输出
@node.define()
def process(data, threshold):
    return [x for x in data if x > threshold]

# ✗ 闭包变量不在缓存键中
def make_processor(factor):
    @node.define()
    def process(x):
        return x * factor  # factor 变化不会触发重算！
    return process
```

**禁用缓存**

```python
@node.define(cache=False)
def get_current_time():
    return datetime.now()
```

**多维节点缓存行为**

当多维广播会展开出大量子节点时，框架使用统一缓存语义：`cache` 同时作用于
当前节点和其广播子节点，不再提供单独的聚合缓存开关。

**排除参数**

```python
@node.define(ignore=["debug", "verbose"])
def compute(x, debug=False, verbose=False):
    if debug: print(f"Computing {x}")
    return x * 2
```

**缓存后端**

| 类型         | 用途           | 配置                       |
| ------------ | -------------- | -------------------------- |
| `MemoryLRU`  | 热数据快速访问 | `maxsize`: 默认 512        |
| `DiskCache` | 冷数据持久化   | `root`: 默认 ".cache"      |
| `ChainCache` | 多级组合       | 按顺序查找，命中后回填上级 |

---

## 追溯

```python
print(repr(square(add(2, 3))))
# 输出:
# hash = 7c3b8fad541e11
# add_0 = add(x=2, y=3)
# square_0 = square(z=add_0)
```

使用 `DiskCache` 时，脚本自动保存：

```
.cache/
├── square/
│   ├── 7c3b8fad541e11.pkl    # 计算结果
│   └── 7c3b8fad541e11.py     # 复现脚本
```

---

## 调度

未缓存的节点可并行执行：

```python
node.configure(executor="thread", workers=4)

# d1, d2, d3 没有相互依赖，可以并行执行
d1, d2, d3 = download(url1), download(url2), download(url3)
```

| 类型        | 适用场景                         |
| ----------- | -------------------------------- |
| `"thread"`  | I/O 密集型（网络请求、文件读写） |
| `"process"` | CPU 密集型（计算、数据处理）     |

**节点级并发控制**

| 配置         | 行为                        |
| ------------ | --------------------------- |
| 不指定       | 使用全局 `workers` 设置     |
| `workers=2`  | 该函数最多 2 个实例同时运行 |
| `workers=-1` | 使用全部 CPU 核心           |
| `local=True` | 在主线程执行，不进入线程池  |

---

## 错误处理

默认单个节点失败不会阻止其他节点执行：

```python
node.configure(continue_on_error=True)  # 默认值

a = may_fail(-1)  # 失败
b = may_fail(1)   # 成功
result = downstream(a, b)  # 被跳过（因为依赖 a）
```

设置 `continue_on_error=False` 可在首个节点失败时立即终止。

---

## 配置

**加载配置**

```python
node.configure(config="config.yaml")
```

```yaml
# config.yaml
load_data:
  path: "data.csv"
  threshold: 0.5
```

**注入默认参数**

```python
@node.define()
def load_data(path, threshold):
    df = pd.read_csv(path)
    return df[df.value > threshold]

result = load_data()()        # 使用配置值
result = load_data(path="other.csv")()  # 调用时覆盖
```

**修改默认参数**

```python
node.cfg.load_data.path = "custom.csv"
load_data()()  # 使用修改后的值
```

**变量引用**

```yaml
base_dir: "/data"
raw_path: "${base_dir}/raw.csv"       # → "/data/raw.csv"
```

**节点依赖**

```yaml
load_data:
  _target_: mymodule.load_data
  path: "data.csv"

process:
  _target_: mymodule.process
  data: ${load_data}  # 自动构建 load_data 节点并注入
```

**预设**

```yaml
train:
  learning_rate: 0.01
  epochs: 100
  _use_: dev
  _presets_:
    dev:  { epochs: 10 }
    prod: { epochs: 1000 }
```

```python
train()()  # epochs=10

node.cfg.train._use_ = "prod"
train()()  # epochs=1000
```

---

## 多维计算

### 维度/坐标

```python
@node.dimension()
def time():
    return [2020, 2021, 2022]
```

### 广播

```python
@node.define()
def load(t):
    return pd.read_csv(f"{t}.csv")

data = load(t=time())()  # 对 3 个时间执行 3 次
```

**多维广播**（笛卡尔积）

```python
@node.dimension()
def model():
    return ["LR", "RF"]

# 3 个时间 × 2 个模型 = 6 个独立计算
grid = train(t=time(), m=model())
```

**维度对齐**：共享维度时自动 zip

```python
t = time()
s1 = step1(t=t)
s2 = step2(x=s1, t=t)  # step1 和 step2 共享 time 维度
```

### 结果访问

返回 `DimensionedResult`（继承自 `numpy.ndarray`）：

```python
result = load(t=time())()

result.dims    # ("time",)
result.coords  # {"time": [2020, 2021, 2022]}
result.shape   # (3,)

result[0]      # 第一个元素
result[-1]     # 最后一个元素

# 按坐标查找
idx = result.coords["time"].index(2021)
result[idx]

# 多维操作
grid.transpose("time", "model")

# 带坐标遍历
for df, coords in result.items():
    df["year"] = coords["time"]
```

### 聚合

**全归约**

```python
@node.define(reduce_dims="all")
def summary(data):
    return {"count": len(data.flat), "total": sum(d["value"] for d in data.flat)}
```

`reduce_dims` 节点收到的 `data` 是 `DimensionedResult`。当声明
`reduce_dims=["time", "model"]` 时，`data.dims` 会严格等于
`("time", "model")`，轴顺序与声明一致。

**部分归约**

```python
@node.define(reduce_dims=["time"])
def site_avg(data):
    return sum(d["value"] for d in data.flat) / len(data.flat)

grid = measure(t=time(), s=site())  # dims=("site", "time")
avgs = site_avg(data=grid)          # dims=("site",)
```

---

## API 参考

### node.configure()

| 参数                | 默认值                                    | 说明                        |
| ------------------- | ----------------------------------------- | --------------------------- |
| `config`            | `None`                                    | Config 对象或 YAML 文件路径 |
| `cache`             | `ChainCache([MemoryLRU(), DiskCache()])` | 缓存后端                    |
| `executor`          | `"thread"`                                | `"thread"` 或 `"process"`   |
| `workers`           | `4`                                       | 默认并发数                  |
| `continue_on_error` | `True`                                    | 节点失败时是否继续          |
| `validate`          | `True`                                    | 是否 Pydantic 验证参数      |

### @node.define()

| 参数          | 默认值   | 说明                      |
| ------------- | -------- | ------------------------- |
| `cache`       | `True`   | 是否缓存当前节点及其广播子节点结果 |
| `workers`     | 继承全局 | 最大并发数，`-1` 全部 CPU |
| `local`       | `False`  | 是否在主线程执行          |
| `reduce_dims` | `()`     | 归约维度，`"all"` 全部    |
| `ignore`      | `None`   | 不计入缓存键的参数名      |

### @node.dimension()

| 参数   | 默认值 | 说明     |
| ------ | ------ | -------- |
| `name` | 函数名 | 维度名称 |

### Node

| 方法                | 说明                     |
| ------------------- | ------------------------ |
| `node()`            | 执行 DAG 并返回结果      |
| `node(force=True)`  | 清除缓存后重新执行       |
| `node.invalidate()` | 清除该节点的缓存         |
| `repr(node)`        | 生成可复现的 Python 脚本 |
| `node.dims`         | 维度名元组               |
| `node.coords`       | 坐标字典                 |

### DimensionedResult

| 属性/方法           | 说明                       |
| ------------------- | -------------------------- |
| `.dims`             | 维度名元组                 |
| `.coords`           | 坐标字典                   |
| `.transpose(*dims)` | 按维度名转置               |
| `.items()`          | 迭代 `(元素, 坐标字典)` 对 |
