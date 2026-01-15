# RFC: Non-Dimension Coordinates

**Status**: Draft  
**Created**: 2026-01-15  

## 动机

当需要遍历多维空间的稀疏子集（流形）时，当前设计需要将多维参数打包为一维 `config` 维度，导致：
1. 丢失原始维度语义
2. 后续 groupby/sel 操作需要手动筛选

## 设计草案

### 坐标类型

| 类型       | 存储格式                    | 示例                                     |
| ---------- | --------------------------- | ---------------------------------------- |
| 维度坐标   | `"dim": [values]`           | `"time": [1, 2, 3]`                      |
| 非维度坐标 | `"name": ("dim", [values])` | `"site": ("sample", ["BJ", "SH", "BJ"])` |
| 0维坐标    | `"name": scalar`            | `"experiment_id": "exp001"`              |


### 待定义的语义

1. **广播传递**：非维度坐标如何在节点间传递？
2. **归约聚合**：归约维度时，非维度坐标如何处理？
3. **声明方式**：`@dimension` 装饰器如何声明非维度坐标？


## 参考

- xarray DataArray: https://docs.xarray.dev/en/stable/user-guide/data-structures.html
