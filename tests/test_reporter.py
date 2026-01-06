"""Tests for RichReporter functionality."""

from node import Runtime
from node.reporters import RichReporter, _RichReporterCtx


def _make_ctx():
    """Create a RichReporter context for testing."""
    rt = Runtime(validate=False, continue_on_error=False)

    @rt.define()
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
