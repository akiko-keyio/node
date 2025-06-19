from __future__ import annotations
# coverage: ignore-file

from typing import Dict, TYPE_CHECKING
import time
import sys
import threading
from queue import SimpleQueue, Empty

from rich.live import Live  # type: ignore[import]
from rich.console import Group  # type: ignore[import]
from rich.text import Text  # type: ignore[import]
from rich.spinner import Spinner  # type: ignore[import]

from .node import Node, _render_call

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from .node import Engine

__all__ = ["RichReporter"]


class RichReporter:
    """Display execution status using ``rich`` in real time."""

    def __init__(self, refresh_per_second: int = 20, show_script_line: bool = True):
        """Create reporter.

        Parameters
        ----------
        refresh_per_second:
            UI refresh rate.
        show_script_line:
            Display canonical script line instead of ``_render_call``.
        """

        self.refresh_per_second = refresh_per_second
        self.show_script_line = show_script_line

    def attach(self, engine: "Engine", root: Node):
        return _RichReporterCtx(self, engine, root)


class _RichReporterCtx:
    def __init__(self, reporter: RichReporter, engine: "Engine", root: Node):
        self.cfg = reporter
        self.engine = engine
        self.root = root
        self.q: SimpleQueue = SimpleQueue()
        self.running: Dict[str, tuple[str, float]] = {}
        self.hits = 0
        self.hit_time = 0.0
        self.execs = 0
        self.exec_start: float | None = None
        self.exec_end: float | None = None
        self.root_info: tuple[str, float, bool] | None = None
        self.spinner = Spinner("dots")

    # --------------------------------------------------------------
    def __enter__(self):
        self.orig_start = self.engine.on_node_start
        self.orig_end = self.engine.on_node_end
        self.orig_flow = self.engine.on_flow_end
        self.engine.on_node_start = self._start
        self.engine.on_node_end = self._end
        self.engine.on_flow_end = self._flow
        self.total = len(self.root.order)
        self.live = Live(self._render(), refresh_per_second=self.cfg.refresh_per_second)
        self.live.__enter__()
        self._stop = threading.Event()
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self.t.join()
        self._drain()
        final_render = self._render(final=True)
        self.live.update(final_render)
        self.live.__exit__(exc_type, exc, tb)
        if "ipykernel" in sys.modules:
            self.live.console.print(final_render)
        self.engine.on_node_start = self.orig_start
        self.engine.on_node_end = self.orig_end
        self.engine.on_flow_end = self.orig_flow

    # --------------------------------------------------------------
    def _start(self, n: Node) -> None:
        if self.cfg.show_script_line:
            call = n.lines[-1][-1]
        else:
            call = _render_call(n.fn, n.args, n.kwargs)
        self.q.put(("start", n.key, call, time.perf_counter()))
        if self.orig_start:
            self.orig_start(n)

    def _end(self, n: Node, dur: float, cached: bool) -> None:
        self.q.put(("end", n.key, dur, cached))
        if self.orig_end:
            self.orig_end(n, dur, cached)

    def _flow(self, root: Node, wall: float, count: int) -> None:
        self.q.put(("flow", wall))
        if self.orig_flow:
            self.orig_flow(root, wall, count)

    # --------------------------------------------------------------
    def _loop(self) -> None:
        sleep = 1.0 / self.cfg.refresh_per_second
        while not self._stop.is_set():
            self._drain()
            self.live.update(self._render())
            time.sleep(sleep)
        self._drain()

    def _drain(self) -> None:
        while True:
            try:
                typ, *rest = self.q.get_nowait()
            except Empty:
                break
            if typ == "start":
                k, label, ts = rest
                self.running[k] = (label, ts)
            elif typ == "end":
                k, dur, cached = rest
                label, ts = self.running.pop(k, (k, time.perf_counter()))
                if cached:
                    self.hits += 1
                    self.hit_time += dur
                else:
                    self.execs += 1
                    if self.exec_start is None or ts < self.exec_start:
                        self.exec_start = ts
                    end = ts + dur
                    if self.exec_end is None or end > self.exec_end:
                        self.exec_end = end
                if k == self.root.key:
                    self.root_info = (label, dur, cached)
            elif typ == "flow":
                wall = rest[0]
                if self.exec_start is None:
                    now = time.perf_counter()
                    self.exec_start = now - wall
                    self.exec_end = now
                elif self.exec_end is None:
                    self.exec_end = self.exec_start + wall

    # --------------------------------------------------------------
    def _header(self, final: bool) -> Text:
        exec_time = 0.0
        if self.exec_start is not None:
            end = (
                self.exec_end
                if final and self.exec_end is not None
                else time.perf_counter()
            )
            exec_time = end - self.exec_start
        done = self.hits + self.execs
        remain = self.total - done - len(self.running)
        avg = (exec_time) / self.execs if self.execs else 0.0
        eta = int(remain * avg)
        parts = [
            ("⚡ Cache ", "bold"),
            (f"{self.hits} "),
            (f"[{self.hit_time:.1f}s]", "gray50"),
            ("\t⭐ Create ", "bold"),
            (f"{int(self.execs)} "),
            (f"[{exec_time:.1f}s]", "gray50"),
        ]
        if not final:
            parts += [
                ("\t📋 Queue ", "bold"),
                (f"{remain} "),
                (f"[ETA: {eta}s]", "gray50"),
            ]
        return Text.assemble(*parts)

    def _render(self, final: bool = False) -> Group:
        out = [self._header(final)]
        now = time.perf_counter()
        icon = str(self.spinner.render(now))
        for label, ts in list(self.running.values()):
            out.append(Text(f"{icon} {label} [{now - ts:.1f}s]"))
        return Group(*out)
