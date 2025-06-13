"""示例脚本
================

本脚本展示如何使用 Node 库构建简单的 DAG、
从 YAML 加载默认参数以及查看磁盘缓存位置。
"""

from __future__ import annotations


import hashlib

import yaml

from node.node import ChainCache, Config, DiskJoblib, Flow, MemoryLRU


yaml_text = """
add:
  y: 5
"""

flow = Flow(
    cache=ChainCache([MemoryLRU(), DiskJoblib(".cache")]),
    config=Config(yaml.safe_load(yaml_text)),
)


@flow.task()
def add(x: int, y: int) -> int:
    return x + y


@flow.task()
def square(z: int) -> int:
    return z * z


@flow.task()
def inc(x: int) -> int:
    return x + 1


def main() -> None:
    node = square(add(square(square(2)), y=square(square(2))))
    result = flow.run(node)
    print(node, result)

    result_cfg = flow.run(add(x=2))
    print("from config:", result_cfg)

    n = inc(3)
    print("cached:", flow.run(n))

    # 查看生成的缓存文件名
    disk = next(c for c in flow.engine.cache.caches if isinstance(c, DiskJoblib))

    p = disk._expr_path(n.signature)
    if not p.exists():
        p = disk._hash_path(n.signature)
    print("file saved as", p)



if __name__ == "__main__":
    main()

