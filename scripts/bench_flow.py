"""Tutorial & benchmark for Node."""
import random
import time
from typing import Any, Tuple

import node
from node import Config


def build_graph() -> Tuple[Any, Any]:
    """Build a 400x400 node grid DAG."""

    node.configure(config=Config(), executor="thread", workers=8)

    @node.define()
    def slow_mul(a, b) -> int:
        time.sleep(random.uniform(0.0, 2.0))
        return a * b

    t0 = time.perf_counter()
    N = 400
    print("Starting building graph")
    grid: list[list[Any]] = [[None] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j == 0:
                grid[i][j] = slow_mul(1, 1)
            elif i == 0:
                grid[i][j] = slow_mul(grid[i][j - 1], 1.01)
            elif j == 0:
                grid[i][j] = slow_mul(grid[i - 1][j], 1)
            else:
                grid[i][j] = slow_mul(grid[i - 1][j], 0.999)

    t1 = time.perf_counter()
    print(f"Building Graph : {t1 - t0:6.2f} s")
    return grid[-1][-1]


def bench() -> None:
    node.reset()
    root = build_graph()

    t0 = time.perf_counter()
    # repr(root) # Evaluating repr for 160000 nodes is very slow and might hit recursion limits differently
    t1 = time.perf_counter()
    print(f"repr : {t1 - t0:6.2f} s (skipped)")

    t0 = time.perf_counter()
    node.run(root)
    t1 = time.perf_counter()
    print(f"cold run : {t1 - t0:6.2f} s")

    t0 = time.perf_counter()
    node.run(root)
    t1 = time.perf_counter()
    print(f"warm run : {t1 - t0:6.2f} s")


if __name__ == "__main__":
    bench()
