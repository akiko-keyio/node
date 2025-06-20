import inspect


def test_render_call_uses_bound_args(flow_factory, monkeypatch):
    flow = flow_factory()

    @flow.node()
    def add(x, y):
        return x + y

    root = add(1, 2)
    # patch after node creation so __init__ already bound signature
    calls = 0
    original = inspect.signature

    def wrapper(fn):
        nonlocal calls
        calls += 1
        return original(fn)

    monkeypatch.setattr(inspect, "signature", wrapper)
    _ = root.lines
    assert calls == 0
