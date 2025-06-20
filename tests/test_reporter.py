from node.node import Flow
from node.reporters import RichReporter


def _make_ctx(hits=0, execs=0):
    flow = Flow()

    @flow.node()
    def dummy():
        return None

    root = dummy()
    ctx = RichReporter().attach(flow.engine, root)
    ctx.total = 1
    ctx.hits = hits
    ctx.execs = execs
    return ctx


def test_header_omits_cache_when_zero():
    ctx = _make_ctx(hits=0, execs=1)
    header = ctx._header(final=False).plain
    assert "⚡️Cache" not in header
    assert "✨️Create" in header


def test_header_omits_create_when_zero():
    ctx = _make_ctx(hits=1, execs=0)
    header = ctx._header(final=False).plain
    assert "✨️Create" not in header
    assert "⚡️Cache" in header


def test_format_duration():
    ctx = _make_ctx()
    assert ctx._format_dur(5) == "5.0s"
    assert ctx._format_dur(65) == "1m 5s"
    assert ctx._format_dur(3661) == "1h 1m 1s"
