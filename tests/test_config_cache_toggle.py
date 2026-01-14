from node import Config


def test_config_cache_toggle(runtime_factory):
    rt = runtime_factory()

    @rt.define()
    def echo(value: int) -> int:
        return value

    # start with value 1
    rt.config = Config(
        {"echo": {"_target_": f"{__name__}.echo", "value": 1}}, cache_nodes=False
    )

    first = echo()
    assert first() == 1

    # modify configuration
    rt.config._conf.echo.value = 2
    second = echo()
    assert second() == 2
    assert first is not second
