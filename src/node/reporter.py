"""Rich-based progress reporter for DAG execution."""

from __future__ import annotations

# coverage: ignore-file

import sys
import threading
import time
from collections.abc import Generator, Iterable
from contextlib import nullcontext
from queue import Empty, SimpleQueue
from typing import TYPE_CHECKING, Any

from rich.console import Console, Group  # type: ignore[import]
from rich.live import Live  # type: ignore[import]
from rich.progress import (  # type: ignore[import]
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.spinner import Spinner  # type: ignore[import]
from rich.syntax import Syntax  # type: ignore[import]
from rich.text import Text  # type: ignore[import]

from .core import Node, _render_call, build_graph
from .logger import console as _console

if TYPE_CHECKING:
    from .runtime import Runtime

__all__ = ["RichReporter", "track"]

_track_ctx = threading.local()
IN_JUPYTER = "ipykernel" in sys.modules


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class RichReporter:
    """Live progress display backed by Rich."""

    def __init__(
        self,
        refresh_per_second: int = 20,
        show_script_line: bool = True,
        *,
        console: Console | None = None,
        force_terminal: bool = False,
    ):
        self.refresh_per_second = refresh_per_second
        self.show_script_line = show_script_line
        if console is not None:
            self.console = console
        elif force_terminal:
            self.console = Console(force_terminal=True)
        else:
            self.console = _console

    def attach(
        self, runtime: Runtime, root: Node, *, order: list[Node] | None = None
    ):
        if getattr(self.console, "_live", None) is not None:
            return nullcontext()
        return _ReporterCtx(self, runtime, root, order=order)


def track(
    sequence: Iterable[Any],
    description: str = "Working...",
    *,
    total: int | None = None,
) -> Generator[Any, None, None]:
    """Yield items from *sequence* with a progress bar.

    When called inside a running node, the bar integrates with the
    reporter display; otherwise falls back to ``rich.progress.track``.
    """
    ctx = getattr(_track_ctx, "ctx", None)
    if ctx is not None:
        yield from ctx.track(sequence, description, total)
        return
    from rich.progress import track as _track

    yield from _track(sequence, description=description, total=total)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

class _Stats:
    __slots__ = ("total", "completed", "running", "first_start", "last_end",
                 "state_counts")

    def __init__(self) -> None:
        self.total = 0
        self.completed = 0
        self.running = 0
        self.first_start: float | None = None
        self.last_end: float | None = None
        self.state_counts: dict[str, int] = {}


class _ReporterCtx:
    _STATE_LABELS = {
        "cache_reading": "reading",
        "executing": "executing",
        "cache_writing": "writing",
    }
    _STATE_ORDER = ("executing", "cache_writing", "cache_reading")
    _LAST_ACTIVE_TTL = 1.0

    def __init__(
        self,
        cfg: RichReporter,
        runtime: Runtime,
        root: Node,
        *,
        order: list[Node] | None = None,
    ):
        self.cfg = cfg
        self.runtime = runtime
        self.root = root
        self._order = order
        self.q: SimpleQueue = SimpleQueue()

        self.stats: dict[str, _Stats] = {}
        self.task_order: list[str] = []

        self.spinner = Spinner("dots")
        self.current_node: int | None = None

        self.node_fn: dict[int, str] = {}
        self.node_states: dict[int, str] = {}
        self.last_active: tuple[str, str, float] | None = None

        self.tracks: dict[str, tuple[Progress, int, int]] = {}
        self.node_tracks: dict[int, set[str]] = {}

        enc = getattr(cfg.console.file, "encoding", None) or "utf-8"
        try:
            "•".encode(enc)
            self._done_char = "•"
        except Exception:
            self._done_char = "*"

    # -- State helpers --------------------------------------------------------

    def _adjust_state(self, node_key: int, new_state: str | None) -> None:
        """Transition *node_key* to *new_state* (or remove if ``None``)."""
        fn = self.node_fn.get(node_key)
        if fn is None:
            return
        st = self.stats[fn]
        prev = self.node_states.pop(node_key, None) if new_state is None \
            else self.node_states.get(node_key)

        if prev is not None and prev != new_state:
            n = st.state_counts.get(prev, 0) - 1
            if n > 0:
                st.state_counts[prev] = n
            else:
                st.state_counts.pop(prev, None)

        if new_state is not None and new_state != prev:
            self.node_states[node_key] = new_state
            st.state_counts[new_state] = st.state_counts.get(new_state, 0) + 1
            if new_state in self._STATE_ORDER:
                label = self._STATE_LABELS.get(new_state, new_state.replace("_", " "))
                self.last_active = (fn, label, time.perf_counter())

    # -- Context manager ------------------------------------------------------

    def __enter__(self):
        self._orig = (
            self.runtime.on_node_start,
            self.runtime.on_node_end,
            getattr(self.runtime, "on_node_state", None),
            self.runtime.on_flow_end,
        )
        self.runtime.on_node_start = self._on_start
        self.runtime.on_node_end = self._on_end
        self.runtime.on_node_state = self._on_state
        self.runtime.on_flow_end = self._on_flow

        order = self._order
        if order is None:
            order, _ = build_graph(self.root, self.runtime.cache)
        for node in order:
            fn = node.fn.__name__
            self.node_fn[node._hash] = fn
            if fn not in self.stats:
                self.stats[fn] = _Stats()
                self.task_order.append(fn)
            self.stats[fn].total += 1

        self.live = Live(
            self._render(),
            refresh_per_second=self.cfg.refresh_per_second,
            transient=not IN_JUPYTER,
            console=self.cfg.console,
        )
        self.live.__enter__()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._stop.set()
        self._thread.join()
        self._drain()
        final = self._render(final=True)
        self.live.update(final, refresh=True)
        self.live.__exit__(*exc_info)
        if not IN_JUPYTER:
            self.live.console.print(final)
        (
            self.runtime.on_node_start,
            self.runtime.on_node_end,
            self.runtime.on_node_state,
            self.runtime.on_flow_end,
        ) = self._orig

    # -- Callbacks ------------------------------------------------------------

    def _on_start(self, n: Node) -> None:
        call = n.script_lines[-1][-1] if self.cfg.show_script_line \
            else _render_call(n.fn, n.inputs)
        label = Syntax(
            call, "python",
            theme="abap" if IN_JUPYTER else "ansi_dark",
            background_color="default",
        ).highlight(call)
        label.rstrip()
        self.q.put(("S", n._hash, label, time.perf_counter()))
        _track_ctx.ctx = self
        _track_ctx.node = n._hash
        self.current_node = n._hash
        if self._orig[0]:
            self._orig[0](n)

    def _on_end(self, n: Node, dur: float, cached: bool, failed: bool) -> None:
        self.q.put(("E", n._hash, dur, cached, failed))
        if getattr(_track_ctx, "ctx", None) is self:
            _track_ctx.ctx = _track_ctx.node = None
        ids = self.node_tracks.get(n._hash)
        if ids:
            for tid in list(ids):
                self.q.put(("TE", tid))
        self.current_node = None
        if self._orig[1]:
            self._orig[1](n, dur, cached, failed)

    def _on_state(self, n: Node, state: str) -> None:
        self.q.put(("T", n._hash, state))
        if self._orig[2]:
            self._orig[2](n, state)

    def _on_flow(self, root: Node, wall: float, count: int, fails: int) -> None:
        self.q.put(("F", wall))
        if self._orig[3]:
            self._orig[3](root, wall, count, fails)

    # -- Event loop -----------------------------------------------------------

    def _loop(self) -> None:
        interval = 1.0 / self.cfg.refresh_per_second
        while not self._stop.is_set():
            self._drain()
            self.live.update(self._render())
            time.sleep(interval)
        self._drain()

    def track(
        self, sequence: Iterable[Any], description: str, total: int | None
    ) -> Generator[Any, None, None]:
        tid = f"{time.perf_counter()}-{threading.get_ident()}"
        self.q.put(("TS", tid, self.current_node or 0, description, total))
        count = 0
        for item in sequence:
            yield item
            count += 1
            self.q.put(("TU", tid, count))
        self.q.put(("TE", tid))

    def _drain(self) -> None:
        while True:
            try:
                msg = self.q.get_nowait()
            except Empty:
                break
            kind = msg[0]
            if kind == "S":
                _, k, _label, ts = msg
                fn = self.node_fn.get(k)
                if fn:
                    st = self.stats[fn]
                    if st.first_start is None:
                        st.first_start = ts
                    st.running += 1
            elif kind == "T":
                self._adjust_state(msg[1], msg[2])
            elif kind == "E":
                _, k, _dur, cached, _failed = msg
                self._adjust_state(k, None)
                fn = self.node_fn.get(k)
                if fn:
                    st = self.stats[fn]
                    st.running = max(0, st.running - 1)
                    if cached:
                        st.total -= 1
                    else:
                        st.completed += 1
                        st.last_end = time.perf_counter()
            elif kind == "TS":
                _, tid, nk, desc, tot = msg
                prog = self._make_progress()
                task = prog.add_task(desc, total=tot)
                self.tracks[tid] = (prog, task, nk)
                if nk:
                    self.node_tracks.setdefault(nk, set()).add(tid)
            elif kind == "TU":
                info = self.tracks.get(msg[1])
                if info:
                    info[0].update(info[1], completed=msg[2])
            elif kind == "TE":
                info = self.tracks.pop(msg[1], None)
                if info and info[2]:
                    ids = self.node_tracks.get(info[2])
                    if ids is not None:
                        ids.discard(msg[1])
                        if not ids:
                            self.node_tracks.pop(info[2], None)

    def _make_progress(self) -> Progress:
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

    # -- Rendering ------------------------------------------------------------

    @staticmethod
    def _estimate_remaining(
        elapsed: float, completed: int, total: int, *, threshold: float = 60.0
    ) -> float | None:
        if elapsed <= threshold or completed <= 0 or total <= completed:
            return None
        return (total - completed) / (completed / elapsed)

    @staticmethod
    def _fmt_dur(s: float) -> str:
        if s >= 3600:
            h, rem = divmod(int(s), 3600)
            m, sec = divmod(rem, 60)
            return f"{h}h {m}m" + (f" {sec}s" if sec else "")
        if s >= 60:
            m, sec = divmod(int(s), 60)
            return f"{m}m {sec}s" if sec else f"{m}m"
        return f"{s:.1f} s"

    @staticmethod
    def _fmt_eta(s: float) -> str:
        if s >= 3600:
            return f"{int(s // 3600)}h"
        if s >= 60:
            return f"{int(s // 60)}m"
        return f"{int(s)}s"

    def _render(self, final: bool = False) -> Group:
        now = time.perf_counter()
        spin = self.spinner.render(now)
        spin.style = "orange1"
        lines: list[Text] = []
        has_active = False

        for fn in self.task_order:
            st = self.stats[fn]
            if st.total <= 0:
                continue
            has_visible_state = any(
                st.state_counts.get(s, 0) > 0 for s in self._STATE_ORDER
            )
            if has_visible_state:
                has_active = True

            done = st.completed >= st.total and st.running == 0
            dur = 0.0
            if st.first_start is not None:
                dur = (st.last_end if done and st.last_end else now) - st.first_start

            if done:
                lines.append(Text.assemble(
                    (self._done_char, "blue"), " ",
                    (str(st.total), "blue"), " ", fn, " ",
                    (f"[{self._fmt_dur(dur)}]", "blue"),
                ))
            elif st.running == 0 and not has_visible_state:
                lines.append(Text.assemble(
                    "~ ", f"{st.completed}/{st.total} ", fn,
                ))
            else:
                parts: list[str | Text | tuple[str, str]] = [
                    spin, " ",
                    (f"{st.completed}/{st.total}", "orange1"), " ", fn,
                ]
                summary = self._state_summary(st)
                if summary:
                    parts += [" ", (f"({summary})", "orange1")]
                eta = self._estimate_remaining(dur, st.completed, st.total)
                time_str = (
                    f"[{self._fmt_dur(dur)} | ETA {self._fmt_eta(eta)}]"
                    if eta else f"[{self._fmt_dur(dur)}]"
                )
                parts += [" ", (time_str, "bold orange1")]
                lines.append(Text.assemble(*parts))

        if not final and lines and not has_active:
            hint = self._last_active_hint(now)
            if hint is not None:
                lines.insert(0, hint)

        return Group(*lines)

    def _state_summary(self, st: _Stats) -> str:
        parts = []
        for state in self._STATE_ORDER:
            c = st.state_counts.get(state, 0)
            if c <= 0:
                continue
            label = self._STATE_LABELS.get(state, state.replace("_", " "))
            parts.append(label if c == 1 else f"{label} x{c}")
        return ", ".join(parts)

    def _last_active_hint(self, now: float) -> Text | None:
        if self.last_active is None:
            return None
        fn, state, ts = self.last_active
        age = now - ts
        if age > self._LAST_ACTIVE_TTL:
            return None
        return Text.assemble(
            ("last active: ", "orange1"),
            (fn, "orange1"),
            (" ", "orange1"),
            (f"({state}, {age:.1f}s ago)", "orange1"),
        )
