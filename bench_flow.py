"""Tutorial & benchmark for Node Flow."""

import time
from typing import Any, Tuple

from src.node import Flow, Config


def build_flow() -> Tuple[Any, Any]:
    """Build a 400x400 node grid DAG."""

    flow = Flow(config=Config(), executor="thread", workers=8)

    @flow.node()
    def slow_mul(a: int, b: int) -> int:
        return a * b

    t0 = time.perf_counter()
    N = 400
    print("Starting building flow")
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
    print(f"Building Flow : {t1 - t0:6.2f} s")
    return flow, grid[-1][-1]


def bench() -> None:
    flow, root = build_flow()

    t0 = time.perf_counter()
    t1 = time.perf_counter()
    print(f"repr : {t1 - t0:6.2f} s")

    t0 = time.perf_counter()
    flow.run(root)
    t1 = time.perf_counter()
    print(f"cold run : {t1 - t0:6.2f} s")

    t0 = time.perf_counter()
    flow.run(root)
    t1 = time.perf_counter()
    print(f"warm run : {t1 - t0:6.2f} s")


if __name__ == "__main__":
    bench()
