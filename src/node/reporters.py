from __future__ import annotations
# coverage: ignore-file

from typing import Dict, Deque, List, TYPE_CHECKING
import time
import threading

from rich.live import Live  # type: ignore[import]
from rich.console import Group  # type: ignore[import]
from rich.text import Text  # type: ignore[import]
from rich.spinner import Spinner  # type: ignore[import]
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.layout import Layout  # type: ignore[import]
from rich.panel import Panel  # type: ignore[import]
from rich.table import Table  # type: ignore[import]
from queue import SimpleQueue, Empty
from collections import deque

from .node import Node, ChainCache, _topo_order

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from .node import Engine

__all__ = ["RichReporter", "SmartReporter"]


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
        for h, line in root.lines:
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

        self.start = time.perf_counter()

        self.progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            transient=True,
            refresh_per_second=self.reporter.refresh_per_second,
        )
        self.bar = self.progress.add_task("task", total=len(self.nodes))

        self.engine.on_node_start = self._on_start
        self.engine.on_node_end = self._on_end

        self.last_done = None
        self.window = 10 + self.engine.workers
        self.truncate = len(self.nodes) > self.window

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
                    self.progress.advance(self.bar)
        self.live.update(self.render(final=True))
        self.live.__exit__(exc_type, exc, tb)
        self.engine.on_node_start = self.orig_start
        self.engine.on_node_end = self.orig_end

    # --------------------------------------------------------------
    def _on_start(self, n: Node):
        mem_hit = disk_hit = False
        if self.caches:
            mem_hit, _ = self.caches[0].get(n.key)
        if not mem_hit and len(self.caches) > 1:
            disk_hit, _ = self.caches[1].get(n.key)
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
        self.last_done = n
        self.progress.advance(self.bar)
        self.live.update(self.render())
        if self.orig_end:
            self.orig_end(n, dur, cached)

    def _refresh_loop(self):
        sleep = 1.0 / self.reporter.refresh_per_second
        while not self._stop_event.is_set():
            self.live.update(self.render())
            time.sleep(sleep)

    # --------------------------------------------------------------
    def _format_line(self, n: Node, frame: Spinner) -> Text:
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
        return Text(f"{icon} {self.labels[n]}{extra}", style=style)

    # --------------------------------------------------------------
    def _window_nodes(self) -> List[Node]:
        nodes = self.nodes
        if not self.truncate:
            return nodes
        idx = {n: i for i, n in enumerate(nodes)}
        running_idx = [idx[n] for n in nodes if self.status[n][0] == "Executing"]
        if running_idx:
            start = max(0, min(running_idx) - 5)
            end = min(len(nodes), max(running_idx) + 6)
        elif self.last_done:
            pos = idx[self.last_done]
            start = max(0, pos - 5)
            end = min(len(nodes), pos + 6)
        else:
            start = 0
            end = self.window
        return nodes[start:end]

    def render(self, final: bool = False) -> Group:
        frame = self.spinner.render(time.perf_counter())
        rows = [self.progress.get_renderable()]

        nodes = self.nodes if final else self._window_nodes()

        rows.extend(self._format_line(n, frame) for n in nodes)
        return Group(*rows)


class SmartReporter:
    """Concurrent-safe reporter with compact UI."""

    def __init__(self, refresh: int = 20, window: int = 20) -> None:
        self.refresh = refresh
        self.window = window

    def attach(self, engine: "Engine", root: Node):
        return _SmartCtx(self, engine, root)


class _SmartCtx:
    def __init__(self, cfg: SmartReporter, engine, root: Node) -> None:
        self.cfg = cfg
        self.engine = engine
        self.root = root
        self.q: SimpleQueue = SimpleQueue()
        self.running: Dict[str, float] = {}
        self.done: Deque[tuple[str, float, bool]] = deque(maxlen=cfg.window)
        self.stats = {"total": len(root.order), "done": 0, "cached": 0, "failed": 0}

    # ------------------------------------------------------------------
    def __enter__(self):
        self.orig_start = self.engine.on_node_start
        self.orig_end = self.engine.on_node_end
        self.engine.on_node_start = self._start
        self.engine.on_node_end = self._end
        self.progress = Progress(
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            transient=False,
            refresh_per_second=self.cfg.refresh,
        )
        self.bar = self.progress.add_task("flow", total=self.stats["total"])
        self.live = Live(self._render(), refresh_per_second=self.cfg.refresh)
        self.live.__enter__()
        self._stop = threading.Event()
        self.t = threading.Thread(target=self._ui_loop, daemon=True)
        self.t.start()
        return self

    # ------------------------------------------------------------------
    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self.t.join()
        self.live.__exit__(exc_type, exc, tb)
        self.engine.on_node_start = self.orig_start
        self.engine.on_node_end = self.orig_end

    # ------------------------------------------------------------------
    def _start(self, n: Node) -> None:
        hit, _ = self.engine.cache.get(n.key)
        self.q.put(("start", n.key, time.perf_counter(), hit))
        if self.orig_start:
            self.orig_start(n)

    def _end(self, n: Node, dur: float, cached: bool) -> None:
        self.q.put(("end", n.key, dur, cached))
        if self.orig_end:
            self.orig_end(n, dur, cached)

    # ------------------------------------------------------------------
    def _ui_loop(self) -> None:
        sleep = 1.0 / self.cfg.refresh
        while not self._stop.is_set():
            self._drain_queue()
            self.live.update(self._render())
            time.sleep(sleep)
        self._drain_queue()
        self.live.update(self._render())

    def _drain_queue(self) -> None:
        while True:
            try:
                event = self.q.get_nowait()
            except Empty:
                break
            if event[0] == "start":
                _, k, ts, hit = event
                if hit:
                    self.stats["cached"] += 1
                    self.stats["done"] += 1
                    self.progress.advance(self.bar)
                else:
                    self.running[k] = ts
            else:
                _, k, dur, cached = event
                self.running.pop(k, None)
                self.done.appendleft((k, dur, cached))
                self.stats["done"] += 1
                self.progress.advance(self.bar)

    # ------------------------------------------------------------------
    def _render(self):
        layout = Layout()
        layout.split(
            Layout(self._make_header(), name="header", size=1),
            Layout(self.progress, name="prog", size=3),
            Layout(name="body"),
        )
        body = Layout()
        body.split_row(
            Layout(self._table_running(), name="running"),
            Layout(self._table_recent(), name="recent"),
        )
        layout["body"].update(body)
        return layout

    def _make_header(self):
        s = self.stats
        return Text.assemble(
            ("✔ ", "green"),
            (f"{s['done']}/{s['total']} "),
            ("⚡ ", "cyan"),
            (str(s["cached"]),),
        )

    def _table_running(self):
        tab = Table(title=f"Running ({len(self.running)})", box=None, expand=True)
        tab.add_column("Node", overflow="fold")
        tab.add_column("Elapsed")
        now = time.perf_counter()
        for k, ts in list(self.running.items())[: self.cfg.window]:
            tab.add_row(k, f"{now - ts:.2f}s")
        return Panel(tab)

    def _table_recent(self):
        tab = Table(title="Recent", box=None, expand=True)
        tab.add_column("Node", overflow="fold")
        tab.add_column("Dur")
        tab.add_column("Type")
        for k, dur, cached in self.done:
            style = "cyan" if cached else "green"
            typ = "cached" if cached else "done"
            tab.add_row(Text(k, style=style), f"{dur:.2f}s", typ)
        return Panel(tab)
