def test_continue_on_error(runtime_factory, capsys):
    rt = runtime_factory()
    rt.continue_on_error = True

    @rt.define()
    def fail():
        raise RuntimeError("boom")

    @rt.define()
    def inc(x):
        return (x or 0) + 1

    result = rt.run(inc(fail()))
    captured = capsys.readouterr().out
    assert result is None
    assert "failed" in captured
