from __future__ import annotations

from typing import Dict, List
from itertools import cycle
import time

from rich.live import Live
from rich.console import Group
from rich.text import Text

from .node import Node, _topo_order, _render_call, ChainCache


class RichReporter:
    """Display execution status using ``rich`` in real time."""

    def __init__(self, refresh_per_second: int = 20):
        self.refresh_per_second = refresh_per_second

    def attach(self, engine: "Engine", root: Node):
        return _RichReporterCtx(self, engine, root)


class _RichReporterCtx:
    def __init__(self, reporter: RichReporter, engine, root: Node):
        self.reporter = reporter
        self.engine = engine
        self.root = root

    # --------------------------------------------------------------
    def _build_lines(self, root: Node):
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
        labels = {n: l for n, l in zip(nodes, lines)}
        return nodes, labels

    # --------------------------------------------------------------
    def __enter__(self):
        self.nodes, self.labels = self._build_lines(self.root)
        self.status: Dict[Node, List] = {n: ["Pending", None, 0.0] for n in self.nodes}

        self.caches = []
        cache = getattr(self.engine, "cache", None)
        if isinstance(cache, ChainCache):
            self.caches = list(cache.caches)

        self.orig_start = self.engine.on_node_start
        self.orig_end = self.engine.on_node_end

        self.spinner = cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])

        self.engine.on_node_start = self._on_start
        self.engine.on_node_end = self._on_end

        self.live = Live(self.render(), refresh_per_second=self.reporter.refresh_per_second)
        self.live.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.live.__exit__(exc_type, exc, tb)
        self.engine.on_node_start = self.orig_start
        self.engine.on_node_end = self.orig_end

    # --------------------------------------------------------------
    def _on_start(self, n: Node):
        mem_hit = disk_hit = False
        if self.caches:
            mem_hit, _ = self.caches[0].get(n.signature)
        if not mem_hit and len(self.caches) > 1:
            disk_hit, _ = self.caches[1].get(n.signature)
        if mem_hit:
            self.status[n][0] = "Cached hit in Memory"
        elif disk_hit:
            self.status[n][0] = "Cached hit in Disk"
        else:
            self.status[n][0] = "Executing"
            self.status[n][1] = time.perf_counter()
        self.live.update(self.render())

    def _on_end(self, n: Node, dur: float, cached: bool):
        if self.status[n][0] not in ("Cached hit in Memory", "Cached hit in Disk"):
            self.status[n][0] = "Executed"
            self.status[n][2] = dur
        self.live.update(self.render())

    # --------------------------------------------------------------
    def render(self) -> Group:
        frame = next(self.spinner)
        rows = []
        for n in self.nodes:
            st, start, dur = self.status[n]
            if st == "Executing":
                icon = frame
                elapsed = time.perf_counter() - (start or time.perf_counter())
                extra = f" [{elapsed:.3f}s]"
                style = ""
            elif st == "Pending":
                icon = "●"
                extra = ""
                style = "yellow"
            else:
                icon = "✔"
                extra = ""
                if st.startswith("Cached hit"):
                    extra = f" ({st.split()[-1].lower()})"
                if dur:
                    extra += f" [{dur:.3f}s]"
                style = "blue"
            rows.append(Text(f"{icon} {self.labels[n]}{extra}", style=style))
        return Group(*rows)
