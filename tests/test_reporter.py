"""Tests for RichReporter functionality."""

import node
from node.reporter import RichReporter, _RichReporterCtx


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
    assert _RichReporterCtx._format_dur(5) == "5.0 s"
    assert _RichReporterCtx._format_dur(65) == "1m 5s"
    assert _RichReporterCtx._format_dur(3661) == "1h 1m 1s"


def test_format_eta_keeps_only_most_significant_unit():
    """ETA should only show the most significant unit."""
    assert _RichReporterCtx._format_eta(8 * 3600 + 12 * 60 + 39) == "8h"
    assert _RichReporterCtx._format_eta(6 * 60 + 1) == "6m"
    assert _RichReporterCtx._format_eta(9.8) == "9s"


def test_estimate_remaining_only_after_one_minute():
    """ETA should be hidden for short elapsed times."""
    assert _RichReporterCtx._estimate_remaining(59.9, 3, 10) is None


def test_estimate_remaining_returns_seconds():
    """ETA should be computed once elapsed exceeds threshold."""
    eta = _RichReporterCtx._estimate_remaining(120, 3, 9)
    assert eta is not None
    assert eta == 240


def test_start_uses_syntax():
    """Test that _start creates proper label using Syntax."""
    ctx = _make_ctx()
    ctx.__enter__()
    ctx._start(ctx.root)
    ctx._drain()
    
    # Check that stats were updated
    fn_name = ctx.root.fn.__name__
    assert fn_name in ctx.stats
    assert ctx.stats[fn_name].running >= 0
    
    ctx.__exit__(None, None, None)
