from __future__ import annotations
# coverage: ignore-file

from typing import Dict, TYPE_CHECKING
from contextlib import nullcontext
import time
import sys
import threading
from queue import SimpleQueue, Empty
from multiprocessing import Queue
from typing import Iterable, Generator, Any
from rich.progress import (
    Progress,
    BarColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)  # type: ignore[import]

from rich.live import Live  # type: ignore[import]
from rich.console import Group, Console  # type: ignore[import]
from rich.text import Text  # type: ignore[import]
from rich.spinner import Spinner  # type: ignore[import]
from rich.syntax import Syntax  # type: ignore[import]
from rich.table import Table  # type: ignore[import]

from .node import Node, _render_call
from .logger import console as _console

# active reporter context for track()
_track_ctx = threading.local()
_process_queue: Queue | None = None


def _set_process_queue(q: Queue | None) -> None:
    global _process_queue
    _process_queue = q
    setattr(_track_ctx, "proc_queue", q)


def _worker_init(q: Queue) -> None:
    _set_process_queue(q)


IN_JUPYTER = "ipykernel" in sys.modules

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from .node import Engine

__all__ = ["RichReporter", "track"]


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
        self.console = console or _console
        if force_terminal:
            # Rich stores the flag on a private attribute; mutate in place to
            # avoid creating additional consoles and keep output unified.
            if hasattr(self.console, "_force_terminal"):
                self.console._force_terminal = True

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
        self.tracks: Dict[str, tuple[Progress, int, str]] = {}
        self.hits = 0
        self.hit_time = 0.0
        self.execs = 0
        self.exec_start: float | None = None
        self.exec_end: float | None = None
        self.spinner = Spinner("dots")
        self.current_node: str | None = None
        self.proc_queue: Queue | None = None
        self.seen: set[str] = set()

    def _make_progress(self) -> Progress:
        """Return a progress instance configured for single-line display."""
        return Progress(
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("left:{task.remaining}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            auto_refresh=False,
            console=self.cfg.console,
            refresh_per_second=self.cfg.refresh_per_second,
        )

    def _get_track(self, node_key: str) -> Progress | None:
        """Return progress object for ``node_key`` if active."""
        for prog, _, node in self.tracks.values():
            if node == node_key:
                return prog
        return None

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
        order = self.root.order
        self.total = len(order)
        self.seen = {n.key for n in order}
        self.live = Live(
            self._render(),
            refresh_per_second=self.cfg.refresh_per_second,
            transient=not IN_JUPYTER,
            console=self.cfg.console,
        )
        self.live.__enter__()
        self.proc_queue = _process_queue
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
        _set_process_queue(None)

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
        _track_ctx.ctx = self
        _track_ctx.node = n.key
        self.current_node = n.key

        if self.orig_start:
            self.orig_start(n)

    def _end(self, n: Node, dur: float, cached: bool) -> None:
        self.q.put(("end", n.key, dur, cached))
        if getattr(_track_ctx, "ctx", None) is self:
            _track_ctx.ctx = None
            _track_ctx.node = None
        for tid, (_, _, node) in list(self.tracks.items()):
            if node == n.key:
                self.q.put(("track_end", tid))
        self.current_node = None
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

    def track(
        self, sequence: Iterable[Any], description: str, total: int | None
    ) -> Generator[Any, None, None]:
        tid = f"{time.perf_counter()}-{threading.get_ident()}"
        self.q.put(("track_start", tid, self.current_node or "", description, total))
        count = 0
        for item in sequence:
            yield item
            count += 1
            self.q.put(("track_update", tid, count))
        self.q.put(("track_end", tid))

    def _drain(self) -> None:
        if self.proc_queue is not None:
            while True:
                try:
                    self.q.put(self.proc_queue.get_nowait())
                except Empty:
                    break
        while True:
            try:
                typ, *rest = self.q.get_nowait()
            except Empty:
                break
            if typ == "start":
                k, label, ts = rest
                self.running[k] = (label, ts)
                if k not in self.seen:
                    self.seen.add(k)
                    self.total += 1
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
            elif typ == "track_start":
                tid, node_key, desc, total = rest
                prog = self._make_progress()
                task = prog.add_task(desc, total=total)
                self.tracks[tid] = (prog, task, node_key)
            elif typ == "track_update":
                tid, completed = rest
                prog, task, _ = self.tracks.get(tid, (None, None, None))
                if prog is not None:
                    prog.update(task, completed=completed)
            elif typ == "track_end":
                tid = rest[0]
                self.tracks.pop(tid, None)
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
        table = Table.grid(expand=True)
        for key, (label, ts) in list(self.running.items()):
            dur = self._format_dur(now - ts)
            row = Text.assemble(icon, " ", label, (f" [{dur}]", "gray50"))
            track = self._get_track(key)
            if track:
                table.add_row(row, track)
            else:
                table.add_row(row)
        out.append(table)
        return Group(*out)


def track(
    sequence: Iterable[Any],
    description: str = "Working...",
    *,
    total: int | None = None,
) -> Generator[Any, None, None]:
    ctx = getattr(_track_ctx, "ctx", None)
    if ctx is not None:
        yield from ctx.track(sequence, description, total)
        return
    proc_q = getattr(_track_ctx, "proc_queue", _process_queue)
    node_key = getattr(_track_ctx, "node", "")
    if proc_q is not None and node_key:
        tid = f"{time.perf_counter()}-{threading.get_ident()}"
        proc_q.put(("track_start", tid, node_key, description, total))
        count = 0
        for item in sequence:
            yield item
            count += 1
            proc_q.put(("track_update", tid, count))
        proc_q.put(("track_end", tid))
    else:
        from rich.progress import track as _track

        yield from _track(
            sequence, description=description, total=total, console=_console
        )
