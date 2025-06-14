from node.node import Flow, _topo_order, ChainCache
from rich.live import Live
from rich.table import Table
import time

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

if __name__ == "__main__":
    root = square(add(square(2), square(2)))

    # Determine execution order and initialize status tracking
    order = _topo_order(root)
    status = {n: ["Pending", 0.0] for n in order}

    def render_table() -> Table:
        table = Table(title="Execution Status")
        table.add_column("Node")
        table.add_column("Status")
        table.add_column("Duration")
        for n in order:
            dur = f"{status[n][1]:.2f}s" if status[n][1] else ""
            table.add_row(n.signature, status[n][0], dur)
        return table

    caches = []
    if isinstance(flow.engine.cache, ChainCache):
        caches = list(flow.engine.cache.caches)

    def on_start(n):
        mem_hit = False
        disk_hit = False
        if caches:
            hit, _ = caches[0].get(n.signature)
            mem_hit = hit
        if not mem_hit and len(caches) > 1:
            hit, _ = caches[1].get(n.signature)
            disk_hit = hit
        if mem_hit:
            status[n][0] = "Cached hit in Memory"
        elif disk_hit:
            status[n][0] = "Cached hit in Disk"
        else:
            status[n][0] = "Executing"
        status[n][1] = 0.0
        live.update(render_table())

    def on_end(n, dur, cached):
        if status[n][0] not in ("Cached hit in Memory", "Cached hit in Disk"):
            status[n][0] = "Executed"
            status[n][1] = dur
        live.update(render_table())

    flow.engine.on_node_start = on_start
    flow.engine.on_node_end = on_end

    with Live(render_table(), refresh_per_second=5) as live:
        result = flow.run(root)

    print("Result:", result)
