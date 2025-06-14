from node.node import Flow
import time
from rich.table import Table
from rich.live import Live

flow = Flow()


@flow.node()
def add(x: int, y: int) -> int:
    time.sleep(2)
    return x + y


@flow.node()
def square(x: int) -> int:
    time.sleep(3)
    return x * x


@flow.node()
def inc(x: int) -> int:
    time.sleep(0.2)
    return x + 1

#
if __name__ == "__main__":
    table = Table(show_header=False)
    table.add_column("node")
    table.add_column("status")

    with Live(table, refresh_per_second=4) as live:
        def log_status(node, dur, cached):
            status = "cached" if cached else f"{dur:.1f}s"
            table.add_row(node.signature, status)
            live.update(table)

        flow.engine.on_node_end = log_status

        node = square(add(square(2), square(2)))
        result = node.get()

    print("Result:", result)
