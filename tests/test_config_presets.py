import pytest

from node.node import Config, Flow


def test_presets_apply_overrides():
    flow = Flow(validate=False)

    @flow.node()
    def add(x: int, y: int, z: int = 0) -> int:
        return x + y + z

    config = Config(
        {
            "baseline": {"value": 5},
            "add": {
                "y": 1,
                "_presets": {
                    "fast": {"y": 2, "z": 3},
                    "ref": {"z": "${baseline.value}"},
                },
            },
        }
    )
    flow.config = config

    assert add(1).get() == 2
    assert add(1, presets="fast").get() == 6
    assert add(1, presets=["fast", "ref"]).get() == 8
    assert add(1, presets=("ref",)).get() == 7


def test_presets_missing_entry_raises():
    flow = Flow(validate=False)

    @flow.node()
    def task(x: int) -> int:
        return x

    flow.config = Config({"task": {"_presets": {"foo": {"x": 1}}}})

    node = task(presets="foo")
    assert node.get() == 1

    with pytest.raises(KeyError, match="Preset 'bar' is not defined"):
        task(presets="bar")
