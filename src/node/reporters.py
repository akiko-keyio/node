from __future__ import annotations
# coverage: ignore-file

from typing import TYPE_CHECKING, Any
from contextlib import nullcontext
import time
import sys
import threading
from queue import SimpleQueue, Empty
from multiprocessing import Queue
from collections.abc import Iterable, Generator
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

from .core import Node, _render_call, _build_graph
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
    from .runtime import Runtime


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
        if console is None:
            if force_terminal:
                self.console = Console(force_terminal=True)
            else:
                self.console = _console
        else:
            self.console = console

    def attach(self, runtime: "Runtime", root: Node):
        """Return a context manager bound to ``engine`` and ``root``.

        If the console already has an active live display, a no-op context
        manager is returned to avoid nested :class:`rich.live.Live` errors.
        """
        if getattr(self.console, "_live", None) is not None:
            return nullcontext()
        return _RichReporterCtx(self, runtime, root)


class _TaskStats:
    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.running = 0
        self.first_start: float | None = None
        self.last_end: float | None = None


class _RichReporterCtx:
    """Context manager handling ``rich`` updates for a run."""

    def __init__(self, reporter: RichReporter, runtime: "Runtime", root: Node):
        self.cfg = reporter
        self.runtime = runtime
        self.root = root
        self.q: SimpleQueue = SimpleQueue()
        
        # Stats by function name
        self.stats: dict[str, _TaskStats] = {}
        # Order of appearance for function names to maintain stable sort
        self.task_order: list[str] = []
        
        self.exec_start: float | None = None
        self.exec_end: float | None = None
        self.spinner = Spinner("dots")
        self.current_node: str | None = None
        self.proc_queue: Queue | None = None
        
        # Internal tracking
        self.node_to_fn_name: dict[str, str] = {}
        self.running_nodes: set[str] = set()
        
        # Restore tracking attributes for track() support
        self.tracks: dict[str, tuple[Progress, int, str]] = {}
        self.node_track_ids: dict[str, set[str]] = {}

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
        return f"{seconds:.1f} s"

    # --------------------------------------------------------------
    def __enter__(self):
        self.orig_start = self.runtime.on_node_start
        self.orig_end = self.runtime.on_node_end
        self.orig_flow = self.runtime.on_flow_end
        self.runtime.on_node_start = self._start
        self.runtime.on_node_end = self._end
        self.runtime.on_flow_end = self._flow
        
        # Pre-calculate totals by function name
        order, _ = _build_graph(self.root, self.runtime.cache)
        for node in order:
            fn_name = node.fn.__name__
            self.node_to_fn_name[node._hash] = fn_name
            if fn_name not in self.stats:
                self.stats[fn_name] = _TaskStats(0)
                self.task_order.append(fn_name)
            self.stats[fn_name].total += 1

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
        if not IN_JUPYTER:
            self.live.console.print(final_render)
        self.runtime.on_node_start = self.orig_start
        self.runtime.on_node_end = self.orig_end
        self.runtime.on_flow_end = self.orig_flow
        _set_process_queue(None)

    # --------------------------------------------------------------
    def _start(self, n: Node) -> None:
        if self.cfg.show_script_line:
            call = n.script_lines[-1][-1]
        else:
            call = _render_call(n.fn, n.inputs)

        label = Syntax(
            call,
            "python",
            theme="abap" if IN_JUPYTER else "ansi_dark",
            background_color="default",
        ).highlight(call)
        label.rstrip()
        self.q.put(("start", n._hash, label, time.perf_counter()))
        _track_ctx.ctx = self
        _track_ctx.node = n._hash
        self.current_node = n._hash

        if self.orig_start:
            self.orig_start(n)

    def _end(self, n: Node, dur: float, cached: bool, failed: bool) -> None:
        self.q.put(("end", n._hash, dur, cached, failed))
        if getattr(_track_ctx, "ctx", None) is self:
            _track_ctx.ctx = None
            _track_ctx.node = None
        ids = self.node_track_ids.get(n._hash)
        if ids:
            for tid in list(ids):
                self.q.put(("track_end", tid))
        self.current_node = None
        if self.orig_end:
            self.orig_end(n, dur, cached, failed)

    def _flow(self, root: Node, wall: float, count: int, fails: int) -> None:
        self.q.put(("flow", wall))
        if self.orig_flow:
            self.orig_flow(root, wall, count, fails)

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
                fn_name = self.node_to_fn_name.get(k)
                if fn_name:
                    stats = self.stats[fn_name]
                    if stats.first_start is None:
                        stats.first_start = ts
                    stats.running += 1
                self.running_nodes.add(k)
                    
            elif typ == "end":
                k, dur, cached, failed = rest
                if k in self.running_nodes:
                    self.running_nodes.remove(k)
                
                fn_name = self.node_to_fn_name.get(k)
                if fn_name:
                    stats = self.stats[fn_name]
                    
                    # Always decrement running on end
                    stats.running -= 1
                    if stats.running < 0:
                        stats.running = 0

                    if cached:
                        # Cached tasks are hidden from the total count
                        stats.total -= 1
                    else:
                        stats.completed += 1
                        stats.last_end = time.perf_counter()

            elif typ == "track_start":
                tid, node_key, desc, total = rest
                prog = self._make_progress()
                task = prog.add_task(desc, total=total)
                self.tracks[tid] = (prog, task, node_key)
                if node_key:
                    self.node_track_ids.setdefault(node_key, set()).add(tid)
            elif typ == "track_update":
                tid, completed = rest
                prog, task, _ = self.tracks.get(tid, (None, None, None))
                if prog is not None:
                    prog.update(task, completed=completed)
            elif typ == "track_end":
                tid = rest[0]
                info = self.tracks.pop(tid, None)
                if info is not None:
                    node_key = info[2]
                    if node_key:
                        ids = self.node_track_ids.get(node_key)
                        if ids is not None:
                            ids.discard(tid)
                            if not ids:
                                self.node_track_ids.pop(node_key, None)


    # --------------------------------------------------------------
    def _render(self, final: bool = False) -> Group:
        lines = []
        now = time.perf_counter()
        # Use orange1 for running spinner
        spinner = self.spinner.render(now)
        spinner.style = "orange1"
        
        for fn_name in self.task_order:
            stats = self.stats[fn_name]
            
            # Skip if no tasks to show (all cached or empty)
            if stats.total <= 0:
                continue
            
            is_done = stats.completed >= stats.total and stats.running == 0
            
            # Calculate Duration
            duration = 0.0
            if stats.first_start is not None:
                if is_done and stats.last_end is not None:
                    duration = stats.last_end - stats.first_start
                else:
                    duration = now - stats.first_start
            
            if is_done:
                # Done: Blue style
                # • 12 Task1 Name [12.2 s]
                icon = Text("•", style="blue")
                count_text = f"{stats.total}"
                dur_str = f"[{self._format_dur(duration)}]"
                
                # Function name: default color (no style), not bold
                # Others: blue
                parts = [
                    icon,
                    Text(" "),
                    Text(f"{count_text}", style="blue"),
                    Text(" "),
                    Text(f"{fn_name}"), # Default style
                    Text(" "),
                    Text(dur_str, style="blue")
                ]
            else:
                # Running: Orange style
                # ⠋ 2/7 Task3 Name [1.9 s]
                icon = spinner
                count_text = f"{stats.completed}/{stats.total}"
                dur_str = f"[{self._format_dur(duration)}]"
                
                # Function name: default color (no style), not bold
                # Others: orange1
                # Time: bold orange1
                parts = [
                    icon,
                    Text(" "),
                    Text(f"{count_text}", style="orange1"),
                    Text(" "),
                    Text(f"{fn_name}"), # Default style
                    Text(" "),
                    Text(dur_str, style="bold orange1")
                ]
            
            lines.append(Text.assemble(*parts))
            
        return Group(*lines)


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
    from rich.progress import track as _track
    yield from _track(sequence, description=description, total=total)
