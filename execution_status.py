from node.node import (
    Flow,
    _topo_order,
    ChainCache,
    _render_call,
)
from rich.live import Live
from rich.console import Group
from rich.text import Text
import time


def _build_lines(root):
    order = _topo_order(root)
    sig2var, mapping, lines, nodes = {}, {}, [], []

    for n in order:
        ignore = getattr(n.fn, "_node_ignore", ())
        key = getattr(n, "signature", None) or _render_call(
            n.fn, n.args, n.kwargs, canonical=True, ignore=ignore
        )

        if key in sig2var:
            mapping[n] = sig2var[key]
            if n is root:
                call = _render_call(
                    n.fn,
                    n.args,
                    n.kwargs,
                    canonical=True,
                    mapping=mapping,
                    ignore=ignore,
                )
                lines.append(call)
                nodes.append(n)
            continue

        var = key if n is root else f"n{len(sig2var)}"
        mapping[n] = var
        if n is not root:
            sig2var[key] = var

        call = _render_call(
            n.fn,
            n.args,
            n.kwargs,
            canonical=True,
            mapping=mapping,
            ignore=ignore,
        )
        lines.append(call if n is root else f"{var} = {call}")
        nodes.append(n)

    lines.reverse()
    nodes.reverse()
    labels = {n: l for n, l in zip(nodes, lines)}
    return nodes, labels


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

    nodes, labels = _build_lines(root)
    status = {n: ["Pending", 0.0] for n in nodes}

    caches = []
    if isinstance(flow.engine.cache, ChainCache):
        caches = list(flow.engine.cache.caches)

    def render() -> Group:
        icons = {
            "Pending": "-",
            "Executing": "⠋",
            "Executed": "✔",
            "Cached hit in Memory": "✔",
            "Cached hit in Disk": "✔",
        }
        rows = []
        for n in nodes:
            st, dur = status[n]
            icon = icons.get(st, "?")
            extra = ""
            if st.startswith("Cached hit"):
                extra = f" ({st.split()[-1].lower()})"
            if dur:
                extra += f" [{dur:.2f}s]"
            rows.append(Text(f"{icon} {labels[n]}{extra}"))
        return Group(*rows)

    def on_start(n):
        mem_hit = False
        disk_hit = False
        if caches:
            mem_hit, _ = caches[0].get(n.signature)
        if not mem_hit and len(caches) > 1:
            disk_hit, _ = caches[1].get(n.signature)
        if mem_hit:
            status[n][0] = "Cached hit in Memory"
        elif disk_hit:
            status[n][0] = "Cached hit in Disk"
        else:
            status[n][0] = "Executing"
        status[n][1] = 0.0
        live.update(render())

    def on_end(n, dur, cached):
        if status[n][0] not in ("Cached hit in Memory", "Cached hit in Disk"):
            status[n][0] = "Executed"
            status[n][1] = dur
        live.update(render())

    flow.engine.on_node_start = on_start
    flow.engine.on_node_end = on_end

    with Live(render(), refresh_per_second=5) as live:
        result = flow.run(root)

    print("Result:", result)
