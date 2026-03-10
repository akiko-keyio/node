# cache.get(...) 耗时分析

## 1. 调用发生在哪、多少次

| 阶段 | 位置 | 次数 | 是否用到返回值里的 value |
|------|------|------|--------------------------|
| 构图 | `core.build_graph()` | 每个被 DFS 访问到的节点 1 次 | **否**，只用 `[0]`（hit） |
| 执行 | `runtime.Runtime._eval_node()` | 每个被执行的节点 1 次 | **是**，需要 value |

大规模 + dimension 时，节点数 N 可能达到 10^5～10^6。  
- 构图阶段：DFS 会访问到大量（甚至全部）节点，即 **N 次 `cache.get()`，且只用 hit/miss**。  
- 执行阶段：对进入 order 的节点再各调一次 `cache.get()`，需要 value。

因此「每个节点一次 cache.get 判断」在大规模下会被放大成几十万～几百万次调用，**单次再慢一点，总时间就会非常可观**。

---

## 2. 单次 cache.get 的耗时从哪来

默认使用的是 `ChainCache([MemoryLRU(), DiskCache()])`，一次 `get(fn_name, hash_value)` 的路径大致是：

### 2.1 ChainCache.get（cache.py 194-201）

```python
with self._lock:                           # ① 全局锁
    for i, c in enumerate(self.caches):
        hit, val = c.get(fn_name, hash_value)  # ② 依次查 Memory → Disk
        if hit:
            for earlier in self.caches[:i]:
                earlier.put(fn_name, hash_value, val)  # ③ 回填上层
            return True, val
return False, None
```

- **① 锁**：每次 get 拿一次全局锁，N 次 get 就是 N 次加锁/解锁，有固定开销。  
- **② 链式查找**：先 MemoryLRU，未命中再 DiskCache。  
- **③ 回填**：若在 Disk 命中，会对 MemoryLRU 做 `put`，多一次写和锁。

### 2.2 MemoryLRU.get（cache.py 74-78）

```python
with self._lock:
    if hash_value in self._lru:
        return True, self._lru[hash_value]
return False, None
```

- 一次锁 + 一次 dict 查找，**命中时非常快**（微秒级）。  
- 未命中时也很快，但会继续走 Disk。

### 2.3 DiskCache.get（cache.py）——主要瓶颈

```python
p = self._path(fn_name, hash_value)   # Path 拼接，便宜
try:
    with p.open("rb") as fh:                  # ④ 打开文件
        return True, pickle.load(fh)          # ⑤ 整文件读 + 反序列化
except FileNotFoundError:
    return False, None
```

- **④ 打开文件**：存在则 open，又是一次 syscall + 可能的内核/文件系统开销。  
- **⑤ 反序列化**：`pickle.load(fh)` 会读完整文件并反序列化整个对象。  
  - 即使返回值在 **build_graph 里被丢弃**（只用 `[0]`），当前实现也 **一定会做「读文件 + 反序列化」**，这是设计上的浪费。

因此：  
- **内存命中**：单次 get 很快。  
- **内存未命中、磁盘命中**：单次 get = open + 读文件 + unpickle，可能几 ms～几十 ms，且与节点数线性叠加。  
- **磁盘未命中**：一次打开失败或存在性检查，仍然有 syscall 成本。

---

## 3. 为什么说「每个节点一次 cache.get 判断」特别耗时

归纳起来有几类原因：

### 3.1 构图阶段做了「为得到 hit 而做的完整加载」

- `build_graph` 里只关心「这个节点有没有缓存」：

  ```python
  hit = cache is not None and node.cache and cache.get(node.fn.__name__, node._hash)[0]
  ```

- 但 `Cache.get` 的语义是「返回 (hit, value)」，没有「只查是否存在」的接口。  
- 所以 DiskCache 每次命中都会：open → **整文件读 + 完整 unpickle**，再把 value 返回；调用方只用 `[0]`，**value 被立刻丢弃**。  
- 在「全缓存、大规模」场景下，构图阶段就会对 **大量节点** 做不必要的磁盘读和反序列化，这是 **单次 get 耗时被放大的直接原因**。

### 3.2 调用次数与节点数线性相关

- dimension 会把一个逻辑步骤展开成大量标量 Node（`_items.flat`），每个都是独立 cache key。  
- DFS 会遍历到这些节点，每个都触发一次 get。  
- 总耗时 ≈ N × (单次 get 成本)，N 大时即使单次 1ms 也会变成几百秒量级。

### 3.3 磁盘与 syscall 放大

- 大量 `open` + 读文件，在机械盘或网络盘上会变成大量随机/小文件 IO，延迟高。  
- 没有「只判断存在」的路径，无法用「仅 stat 或仅查索引」的轻量路径替代「读+反序列化」。

### 3.4 锁与回填

- 每次 get 持锁，串行化所有 cache 访问；get 内部若在 Disk 命中还会对上层做 put，进一步增加锁和计算。  
- 在单线程构图阶段锁竞争不严重，但「每节点一次 get」本身已经很多，锁的固定开销也会被乘上 N。

---

## 4. 结论与优化方向

- **是的**：在大规模 + dimension 下，「每个节点做一次 cache.get(...) 判断」的过程 **确实很耗时**。  
- 主要原因可以概括为：  
  1. **构图阶段** 只需要 hit/miss，却被迫做 **完整读盘 + 反序列化**（当前接口和实现没有「仅判存在」的路径）。  
  2. **调用次数** 与节点数 N 线性相关，N 很大（10^5～10^6）。  
  3. **单次成本** 在磁盘命中时 = open + read + unpickle，在慢盘上尤其高。  
  4. **锁与回填** 等固定开销被调用次数放大。

可行的优化方向包括：

- **为 Cache 增加「仅判存在」的接口**（例如 `contains(fn_name, hash_value) -> bool`），在 `build_graph` 里用该接口代替 `get(..., node._hash)[0]`，避免在构图阶段做任何反序列化。  
- **DiskCache 实现 contains**：用 `Path.exists()` 判断文件是否存在，不 open、不 load。  
- **ChainCache.contains**：沿链依次查各层 contains，命中则返回 True，不加载 value，不回填（或仅在需要时对上层做轻量标记）。  
- 若仍保留「构图时用 get」的兼容路径，可考虑在 DiskCache 中为「只关心 hit」的调用提供可选轻量路径（例如内部 `get(exists_only=True)`），避免无条件 pickle.load。

这样可以在不改变「每个节点判断一次是否缓存」的语义的前提下，把 **单次判断的成本** 从「读文件+反序列化」降为「存在性检查」，从而显著降低大规模任务下 DFS 遍历 + 缓存查询的总耗时。
