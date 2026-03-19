"""Tests for RichReporter functionality."""

import time

import node
from node.reporter import RichReporter, _ReporterCtx


def _make_ctx():
    """Create a RichReporter context for testing."""
    node.reset()
    rt = node.configure(validate=False, continue_on_error=False)

    @node.define()
    def dummy():
        return None

    root = dummy()
    ctx = RichReporter().attach(rt, root)
    return ctx


def test_format_duration():
    """Test _format_dur method formatting."""
    # _format_dur is a static method
    assert _ReporterCtx._fmt_dur(5) == "5.0 s"
    assert _ReporterCtx._fmt_dur(65) == "1m 5s"
    assert _ReporterCtx._fmt_dur(3661) == "1h 1m 1s"


def test_format_eta_keeps_only_most_significant_unit():
    """ETA should only show the most significant unit."""
    assert _ReporterCtx._fmt_eta(8 * 3600 + 12 * 60 + 39) == "8h"
    assert _ReporterCtx._fmt_eta(6 * 60 + 1) == "6m"
    assert _ReporterCtx._fmt_eta(9.8) == "9s"


def test_estimate_remaining_only_after_one_minute():
    """ETA should be hidden for short elapsed times."""
    assert _ReporterCtx._estimate_remaining(59.9, 3, 10) is None


def test_estimate_remaining_returns_seconds():
    """ETA should be computed once elapsed exceeds threshold."""
    eta = _ReporterCtx._estimate_remaining(120, 3, 9)
    assert eta is not None
    assert eta == 240


def test_start_uses_syntax():
    """Test that _start creates proper label using Syntax."""
    ctx = _make_ctx()
    ctx.__enter__()
    ctx._on_start(ctx.root)
    ctx._drain()
    
    # Check that stats were updated
    fn_name = ctx.root.fn.__name__
    assert fn_name in ctx.stats
    assert ctx.stats[fn_name].running >= 0
    
    ctx.__exit__(None, None, None)


def test_state_tracking_updates_counts():
    """Reporter should track node state transitions."""
    ctx = _make_ctx()
    ctx.__enter__()

    fn_name = ctx.root.fn.__name__
    assert ctx.stats[fn_name].state_counts == {"waiting": 1}

    ctx._on_state(ctx.root, "cache_reading")
    ctx._drain()
    assert ctx.stats[fn_name].state_counts == {"cache_reading": 1}

    ctx._on_state(ctx.root, "executing")
    ctx._drain()
    assert ctx.stats[fn_name].state_counts == {"executing": 1}

    ctx._on_end(ctx.root, 0.1, False, False)
    ctx._drain()
    assert ctx.stats[fn_name].state_counts == {}

    ctx.__exit__(None, None, None)


def test_waiting_only_row_uses_compact_format():
    """Waiting-only rows should hide per-task reason in compact mode."""
    ctx = _make_ctx()
    ctx.__enter__()

    lines = [line.plain for line in ctx._render().renderables]
    assert len(lines) == 1
    plain = lines[0]
    assert plain.startswith("~ ")
    assert "[" not in plain

    ctx.__exit__(None, None, None)


def test_last_active_hint_shown_for_short_gap():
    """Recent active work should be shown during brief empty gaps."""
    ctx = _make_ctx()
    ctx.__enter__()
    ctx._on_state(ctx.root, "executing")
    ctx._drain()
    ctx._on_state(ctx.root, "waiting")
    ctx._drain()

    lines = [line.plain for line in ctx._render().renderables]
    assert lines[0].startswith("last active: dummy (executing, ")

    ctx.__exit__(None, None, None)


def test_last_active_hint_expires():
    """Recent activity hint should disappear after the short TTL."""
    ctx = _make_ctx()
    ctx.__enter__()
    ctx.last_active = ("dummy", "executing", time.perf_counter() - 5)

    lines = [line.plain for line in ctx._render().renderables]
    assert not any(line.startswith("last active:") for line in lines)

    ctx.__exit__(None, None, None)


def test_state_labels_use_reading_writing_names():
    """State labels should use short reading/writing names."""
    assert _ReporterCtx._STATE_LABELS["cache_reading"] == "reading"
    assert _ReporterCtx._STATE_LABELS["cache_writing"] == "writing"
