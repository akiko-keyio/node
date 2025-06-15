from __future__ import annotations
# coverage: ignore-file

from typing import Dict, List, TYPE_CHECKING
import time
import threading

from rich.live import Live  # type: ignore[import]
from rich.console import Group  # type: ignore[import]
from rich.text import Text  # type: ignore[import]
from rich.spinner import Spinner  # type: ignore[import]

from .node import Node, ChainCache, _topo_order

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from .node import Engine

__all__ = ["RichReporter"]


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
        h2node = {n._hash: n for n in order}
        nodes: List[Node] = []
        labels: Dict[Node, str] = {}
        for h, line in root.lines():
            n = h2node[h]
            nodes.append(n)
            label = line
            if n is root and "=" in line:
                label = line.split("=", 1)[1].strip()
            labels[n] = label

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

        self.spinner = Spinner("dots")

        self.engine.on_node_start = self._on_start
        self.engine.on_node_end = self._on_end

        self.live = Live(
            self.render(), refresh_per_second=self.reporter.refresh_per_second
        )
        try:
            self.live.console.clear()
        except Exception:
            pass
        self.live.__enter__()
        self._stop_event = threading.Event()
        self._t = threading.Thread(target=self._refresh_loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop_event.set()
        self._t.join()
        if self.engine._exec_count == 0:
            for n in self.nodes:
                if self.status[n][0] == "Pending":
                    self.status[n][0] = "Skipped"
                    self.status[n][2] = 0.0
            self.live.update(self.render())
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
        if self.orig_start:
            self.orig_start(n)

    def _on_end(self, n: Node, dur: float, cached: bool):
        if self.status[n][0] not in ("Cached hit in Memory", "Cached hit in Disk"):
            self.status[n][0] = "Executed"
        self.status[n][2] = dur
        self.live.update(self.render())
        if self.orig_end:
            self.orig_end(n, dur, cached)

    def _refresh_loop(self):
        sleep = 1.0 / self.reporter.refresh_per_second
        while not self._stop_event.is_set():
            self.live.update(self.render())
            time.sleep(sleep)

    # --------------------------------------------------------------
    def render(self) -> Group:
        frame = self.spinner.render(time.perf_counter())
        rows = []
        for n in self.nodes:
            st, start, dur = self.status[n]
            if st == "Executing":
                icon = str(frame)
                elapsed = time.perf_counter() - (start or time.perf_counter())
                extra = f" [{elapsed:.3f}s]"
                style = ""
            elif st == "Pending":
                icon = "●"
                extra = ""
                style = "yellow"
            elif st == "Skipped":
                icon = "●"
                extra = " (skip)"
                style = "grey50"
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
