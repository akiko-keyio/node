def test_continue_on_error(flow_factory, capsys):
    flow = flow_factory()
    flow.engine.continue_on_error = True

    @flow.node()
    def fail():
        raise RuntimeError("boom")

    @flow.node()
    def inc(x):
        return (x or 0) + 1

    result = flow.run(inc(fail()))
    captured = capsys.readouterr().out
    assert result == 1
    assert "failed" in captured
