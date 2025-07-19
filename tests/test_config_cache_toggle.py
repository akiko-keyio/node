from node.node import Config


def test_config_cache_toggle(flow_factory):
    flow = flow_factory()

    @flow.node()
    def echo(value: int) -> int:
        return value

    # start with value 1
    flow.config = Config(
        {"echo": {"_target_": f"{__name__}.echo", "value": 1}}, cache_nodes=False
    )

    first = echo()
    assert first.get() == 1

    # modify configuration
    flow.config._conf.echo.value = 2
    second = echo()
    assert second.get() == 2
    assert first is not second
