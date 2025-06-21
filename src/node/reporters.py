from __future__ import annotations
# coverage: ignore-file

from typing import Dict, TYPE_CHECKING
from contextlib import nullcontext
import time
import sys
import threading
from queue import SimpleQueue, Empty

from rich.live import Live  # type: ignore[import]
from rich.console import Group, Console  # type: ignore[import]
from rich.text import Text  # type: ignore[import]
from rich.spinner import Spinner  # type: ignore[import]
from rich.syntax import Syntax  # type: ignore[import]
from rich.progress import (
    Progress,
    BarColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from .node import Node, _render_call
from .logger import console as _console

IN_JUPYTER = "ipykernel" in sys.modules

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from .node import Engine

__all__ = ["RichReporter"]


class RichReporter:
    """Rich based progress reporter.

    ``RichReporter`` hooks into :class:`~node.node.Engine` callbacks to display
    live execution progress. A separate thread updates a status bar using the
    :mod:`rich` library while nodes are being executed. When running inside
    IPython or Jupyter, the output is kept within the cell; otherwise the final
    status is printed to ``stdout``.
    """

    def __init__(
        self,
        refresh_per_second: int = 20,
        show_script_line: bool = True,
        *,
        console: Console | None = None,
        force_terminal: bool = False,
    ):
        """Create reporter.

        Parameters
        ----------
        refresh_per_second:
            UI refresh rate.
        show_script_line:
            Display canonical script line instead of ``_render_call``.
        force_terminal:
            Force Rich to treat the console as a real terminal.
        """

        self.refresh_per_second = refresh_per_second
        self.show_script_line = show_script_line
        if console is None:
            if force_terminal:
                self.console = Console(force_terminal=True)
            else:
                self.console = _console
        else:
            self.console = console

    def attach(self, engine: "Engine", root: Node):
        """Return a context manager bound to ``engine`` and ``root``.

        If the console already has an active live display, a no-op context
        manager is returned to avoid nested :class:`rich.live.Live` errors.
        """
        if getattr(self.console, "_live", None) is not None:
            return nullcontext()
        return _RichReporterCtx(self, engine, root)


class _RichReporterCtx:
    """Context manager handling ``rich`` updates for a run."""

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
        self.spinner = Spinner("dots")
        self._pause = threading.Event()
        self._progs: list[Progress] = []

    @staticmethod
    def _format_dur(seconds: float) -> str:
        """Return duration string with hours and minutes."""
        if seconds >= 3600:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            parts = [f"{h}h", f"{m}m"]
            if s:
                parts.append(f"{s}s")
            return " ".join(parts)
        if seconds >= 60:
            m = int(seconds // 60)
            s = int(seconds % 60)
            return f"{m}m {s}s" if s else f"{m}m"
        return f"{seconds:.1f}s"

    # --------------------------------------------------------------
    def __enter__(self):
        self.orig_start = self.engine.on_node_start
        self.orig_end = self.engine.on_node_end
        self.orig_flow = self.engine.on_flow_end
        self.engine.on_node_start = self._start
        self.engine.on_node_end = self._end
        self.engine.on_flow_end = self._flow
        self.total = len(self.root.order)
        self.live = Live(
            self._render(),
            refresh_per_second=self.cfg.refresh_per_second,
            transient=not IN_JUPYTER,
            console=self.cfg.console,
        )
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
        self.live.update(final_render, refresh=True)
        self.live.__exit__(exc_type, exc, tb)
        if IN_JUPYTER:
            pass
        else:
            self.live.console.print(final_render)
        self.engine.on_node_start = self.orig_start
        self.engine.on_node_end = self.orig_end
        self.engine.on_flow_end = self.orig_flow

    # --------------------------------------------------------------
    def pause(self):
        """Temporarily release the live console."""
        return _PauseCtx(self)

    def track(
        self,
        sequence,
        description: str = "Working...",
        total: int | None = None,
    ):
        """Iterate over ``sequence`` with an embedded progress bar."""
        columns = [
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(elapsed_when_finished=True),
        ]
        prog = Progress(
            *columns,
            auto_refresh=False,
            console=self.cfg.console,
            transient=False,
        )
        if total is None:
            try:
                total = len(sequence)  # type: ignore[arg-type]
            except Exception:
                total = None
        task = prog.add_task(description, total=total)
        self._progs.append(prog)
        try:
            for item in sequence:
                yield item
                prog.advance(task)
        finally:
            self._progs.remove(prog)

    # --------------------------------------------------------------
    def _start(self, n: Node) -> None:
        if self.cfg.show_script_line:
            call = n.lines[-1][-1]
        else:
            call = _render_call(n.fn, n.args, n.kwargs, bound=n.bound_args)

        label = Syntax(
            call,
            "python",
            theme="abap" if IN_JUPYTER else "ansi_dark",
            background_color="default",
        ).highlight(call)
        label.rstrip()
        self.q.put(("start", n.key, label, time.perf_counter()))

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
            if not self._pause.is_set():
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
        fmt = self._format_dur
        parts = []
        if self.hits:
            parts += [
                ("⚡️"),
                (" Cache ", "bold"),
                (f"{self.hits} ", "bold"),
                (f"[{fmt(self.hit_time)}]", "gray50"),
            ]
        if self.execs:
            prefix = "\t" if parts else ""
            parts += [
                (f"{prefix}✨️"),
                (" Create ", "bold"),
                (f"{int(self.execs)} ", "bold"),
                (f"[{fmt(exec_time)}]", "gray50"),
            ]
        if not final:
            prefix = "\t" if parts else ""
            parts += [
                (f"{prefix}📋️"),
                (" Pending ", "bold"),
                (f"{remain} ", "bold"),
                (f"[ETA: {fmt(eta)}]", "gray50"),
            ]
        return Text.assemble(*parts)

    def _render(self, final: bool = False) -> Group:
        out = [self._header(final)]
        now = time.perf_counter()
        icon = str(self.spinner.render(now))
        for label, ts in list(self.running.values()):
            dur = self._format_dur(now - ts)
            out.append(Text.assemble(icon, " ", label, (f" [{dur}]", "gray50")))
        for prog in self._progs:
            out.append(prog.get_renderable())
        return Group(*out)


class _PauseCtx:
    """Context manager pausing reporter updates."""

    def __init__(self, ctx: _RichReporterCtx):
        self.ctx = ctx

    def __enter__(self):
        self.ctx._pause.set()
        self.ctx.live.stop()

    def __exit__(self, exc_type, exc, tb):
        self.ctx.live.start(refresh=True)
        self.ctx._pause.clear()
