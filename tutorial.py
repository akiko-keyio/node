"""Node usage tutorial
====================

This single script combines the previous ``tutorial.py`` and
``execution_status.py`` examples.  It first walks through a quick start and
then demonstrates advanced features such as ``repr`` generation, caching,
concurrency and live reporting.
"""

from __future__ import annotations

import time
import yaml  # type: ignore[import]

from node.node import ChainCache, Config, DiskJoblib, Flow, MemoryLRU
from node.reporters import RichReporter


# --------------------------------------------------------------
# Quick start
# --------------------------------------------------------------
quick = Flow()


@quick.node()
def add(x: int, y: int) -> int:
    return x + y


@quick.node()
def square(z: int) -> int:
    return z * z


def quick_start() -> None:
    root = square(add(2, 3))
    print("Quick result:", quick.run(root))
    print("repr(root) =", repr(root))


# --------------------------------------------------------------
# Advanced topics
# --------------------------------------------------------------
yaml_text = """
add:
  y: 5
"""

advanced = Flow(
    cache=ChainCache([MemoryLRU(), DiskJoblib(".cache")]),
    config=Config(yaml.safe_load(yaml_text)),
    executor="thread",
    default_workers=2,
)


@advanced.node()
def slow_add(x: int, y: int) -> int:
    time.sleep(2)
    return x + y


@advanced.node()
def slow_square(x: int) -> int:
    time.sleep(3)
    return x * x


@advanced.node()
def inc(x: int) -> int:
    time.sleep(0.2)
    return x + 1


def advanced_topics() -> None:
    root = slow_square(slow_add(slow_square(2), slow_square(2)))
    reporter = RichReporter()
    result = advanced.run(root, reporter=reporter)
    print("Advanced result:", result)
    print("repr(root) =", repr(root))

    start = time.perf_counter()
    advanced.run(root, reporter=reporter)
    elapsed = time.perf_counter() - start
    print(f"Second run from cache: {elapsed:.2f}s")


if __name__ == "__main__":
    quick_start()
    advanced_topics()
