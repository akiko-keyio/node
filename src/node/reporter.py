"""Rich-based progress reporter for DAG execution."""

from __future__ import annotations

# coverage: ignore-file

import io
import sys
import threading
import time
from collections.abc import Generator, Iterable
from contextlib import nullcontext
from queue import Empty, SimpleQueue
from typing import TYPE_CHECKING, Any

from rich.console import Console, ConsoleOptions, Group, RenderResult  # type: ignore[import]
from rich.live import Live  # type: ignore[import]
from rich.spinner import Spinner  # type: ignore[import]
from rich.text import Text  # type: ignore[import]

from .core import Node, build_graph
from .logger import (
    _is_jupyter,
    _remove_rich_handler,
    _restore_rich_handler,
    console as _console,
    logger,
)

if TYPE_CHECKING:
    from .runtime import Runtime

__all__ = ["RichReporter", "track"]

_track_ctx = threading.local()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class RichReporter:
    """Live progress display backed by Rich."""

    def __init__(
        self,
        refresh_per_second: int = 20,
        *,
        console: Console | None = None,
        force_terminal: bool = False,
    ):
        self.refresh_per_second = refresh_per_second
        self._jupyter = _is_jupyter()
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
        yield from sequence
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


class _LiveRenderable:
    """Drains queued events and yields fresh output each time Rich renders."""

    def __init__(self, ctx: _ReporterCtx) -> None:
        self._ctx = ctx

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        self._ctx._drain()
        yield self._ctx._render()


class _ReporterCtx:
    _STATE_LABELS = {
        "cache_reading": "reading",
        "executing": "executing",
        "cache_writing": "writing",
    }
    _STATE_ORDER = ("executing", "cache_writing", "cache_reading")

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
        self._jupyter = False

        enc = getattr(cfg.console.file, "encoding", None) or "utf-8"
        try:
            "•".encode(enc)
            self._done_char = "•"
        except Exception:
            self._done_char = "*"

    # -- State helpers --------------------------------------------------------

    def _queue(self, msg: tuple[Any, ...]) -> None:
        """Push one reporter event (and trigger Jupyter refresh if active)."""
        self.q.put(msg)
        if self._jupyter:
            self._refresh_jupyter()

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

        self._jupyter = self.cfg._jupyter

        for node in order:
            fn = node.fn.__name__
            self.node_fn[node._hash] = fn
            if fn not in self.stats:
                self.stats[fn] = _Stats()
                self.task_order.append(fn)
            self.stats[fn].total += 1
            self._adjust_state(node._hash, "waiting")

        if self._jupyter:
            self._last_refresh = 0.0
            self._refresh_lock = threading.Lock()
            self.cfg.console._live = self
            self._print_jupyter()
        else:
            # Buffer log messages during Live to prevent stdout conflicts.
            self._log_records: list[str] = []
            _remove_rich_handler()
            self._buf_handler_id = logger.add(
                lambda msg: self._log_records.append(str(msg)),
                level="DEBUG",
                format="{time:HH:mm:ss} | {level:<8} | {message}",
                colorize=False,
            )
            self.live = Live(
                _LiveRenderable(self),
                refresh_per_second=self.cfg.refresh_per_second,
                transient=True,
                console=self.cfg.console,
            )
            self.live.__enter__()
        return self

    def __exit__(self, *exc_info):
        self._drain()

        if self._jupyter:
            self._print_jupyter(final=True)
            self.cfg.console._live = None
        else:
            final = self._render(final=True)
            self.live.update(final, refresh=True)
            self.live.__exit__(*exc_info)
            # Restore RichHandler and flush buffered logs after Live area.
            logger.remove(self._buf_handler_id)
            _restore_rich_handler()
            self.cfg.console.print(final)
            for record in self._log_records:
                self.cfg.console.print(record, end="", highlight=False)

        (
            self.runtime.on_node_start,
            self.runtime.on_node_end,
            self.runtime.on_node_state,
            self.runtime.on_flow_end,
        ) = self._orig

    # -- Jupyter refresh ------------------------------------------------------

    def _print_jupyter(self, final: bool = False) -> None:
        """Render progress to stdout with ANSI colors for Jupyter."""
        from IPython.display import clear_output

        clear_output(wait=True)
        buf = io.StringIO()
        Console(
            file=buf, force_terminal=True,
            color_system="truecolor", width=120,
        ).print(self._render(final=final))
        sys.stdout.write(buf.getvalue())
        sys.stdout.flush()

    def _refresh_jupyter(self) -> None:
        """Rate-limited display update for Jupyter, safe from any thread."""
        now = time.monotonic()
        if now - self._last_refresh < 0.5:
            return
        if not self._refresh_lock.acquire(blocking=False):
            return
        try:
            self._last_refresh = now
            self._drain()
            self._print_jupyter()
        except Exception:
            pass
        finally:
            self._refresh_lock.release()

    # -- Callbacks ------------------------------------------------------------

    def _on_start(self, n: Node) -> None:
        self._queue(("S", n._hash, time.perf_counter()))
        _track_ctx.ctx = self
        _track_ctx.node = n._hash
        self.current_node = n._hash
        if self._orig[0]:
            self._orig[0](n)

    def _on_end(self, n: Node, dur: float, cached: bool, failed: bool) -> None:
        self._queue(("E", n._hash, dur, cached, failed))
        if getattr(_track_ctx, "ctx", None) is self:
            _track_ctx.ctx = _track_ctx.node = None
        self.current_node = None
        if self._orig[1]:
            self._orig[1](n, dur, cached, failed)

    def _on_state(self, n: Node, state: str) -> None:
        self._queue(("T", n._hash, state))
        if self._orig[2]:
            self._orig[2](n, state)

    def _on_flow(self, root: Node, wall: float, count: int, fails: int) -> None:
        self._queue(("F", wall))
        if self._orig[3]:
            self._orig[3](root, wall, count, fails)

    # -- Event drain ----------------------------------------------------------

    def _drain(self) -> None:
        while True:
            try:
                msg = self.q.get_nowait()
            except Empty:
                break
            kind = msg[0]
            if kind == "S":
                _, k, ts = msg
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

        for fn in self.task_order:
            st = self.stats[fn]
            if st.total <= 0:
                continue

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
            elif all(
                s == "waiting" or c <= 0
                for s, c in st.state_counts.items()
            ) and st.state_counts.get("waiting", 0) > 0:
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
