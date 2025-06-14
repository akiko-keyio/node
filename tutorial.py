"""示例脚本
================

本脚本展示如何使用 Node 库构建简单的 DAG、
从 YAML 加载默认参数以及查看磁盘缓存位置。
计算结果会以节点的 ``signature`` 字符串作为缓存键。
"""

from __future__ import annotations

import yaml  # type: ignore[import]

from node.node import ChainCache, Config, DiskJoblib, Flow, MemoryLRU

yaml_text = """
add:
  y: 5
"""

flow = Flow(
    cache=ChainCache([MemoryLRU(), DiskJoblib(".cache")]),
    config=Config(yaml.safe_load(yaml_text)),
)


@flow.node()
def add(x: int, y: int) -> int:
    return x + y


@flow.node()
def square(z: int) -> int:
    return z * z


@flow.node()
def inc(x: int) -> int:
    return x + 1


def main() -> None:
    node = square(add(square(2), square(2)))
    print(flow.run(node))
    print(node)
    node = square(add(square(2), square(3)))
    print(flow.run(node))
    print(node)


if __name__ == "__main__":
    main()
