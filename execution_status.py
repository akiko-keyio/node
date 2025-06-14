from node.node import Flow
import time
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from node.node import _topo_order, Engine

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
    console = Console()
    node = square(add(square(2), square(2)))
    order = _topo_order(node)

    progress = Progress(
        TextColumn("{task.fields[signature]}", justify="left"),
        SpinnerColumn(),
        TextColumn("{task.fields[status]}", justify="right"),
        console=console,
        refresh_per_second=5,
    )
    tasks = {n: progress.add_task("", signature=n.signature, status="pending", total=None) for n in order}

    def on_start(n):
        progress.update(tasks[n], status="running")

    def on_end(n, dur, cached):
        status = "cached" if cached else f"{dur:.1f}s"
        progress.update(tasks[n], status=status, completed=1)

    flow.engine.on_node_start = on_start
    flow.engine.on_node_end = on_end

    with progress:
        result = node.get()

    console.print("Result:", result)
