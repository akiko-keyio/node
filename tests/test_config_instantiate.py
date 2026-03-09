import sys

import pytest

import node
from node import Config, ConfigurationError


def test_instantiate_basic(runtime_factory):
    @node.define()
    def add(x: int, y: int) -> int:
        return x + y

    module_name = __name__
    this_module = sys.modules[module_name]
    setattr(this_module, "add", add)

    cfg = {
        "sum_node": {
            "_target_": f"{module_name}.add",
            "x": 2,
            "y": 3,
        }
    }
    runtime_factory(config=Config(cfg))

    n = node.instantiate("sum_node")
    assert n() == 5


def test_instantiate_dependency(runtime_factory):
    @node.define()
    def base(a: int, b: int) -> int:
        return a + b

    @node.define()
    def combine(dep: int, c: int) -> int:
        return dep * c

    module_name = __name__
    this_module = sys.modules[module_name]
    setattr(this_module, "base", base)
    setattr(this_module, "combine", combine)

    cfg = {
        "base_node": {
            "_target_": f"{module_name}.base",
            "a": 2,
            "b": 3,
        },
        "combine_node": {
            "_target_": f"{module_name}.combine",
            "dep": "${base_node}",
            "c": 4,
        },
    }
    runtime_factory(config=Config(cfg))

    n = node.instantiate("combine_node")
    assert n() == 20


def test_instantiate_cache_nodes_toggle(runtime_factory):
    @node.define()
    def echo(value: int) -> int:
        return value

    module_name = __name__
    this_module = sys.modules[module_name]
    setattr(this_module, "echo", echo)

    cfg = {"echo_node": {"_target_": f"{module_name}.echo", "value": 1}}
    rt = runtime_factory(config=Config(cfg, cache_nodes=False))

    n1 = node.instantiate("echo_node")
    rt.config._conf.echo_node.value = 2
    n2 = node.instantiate("echo_node")
    assert n1 is not n2
    assert n2() == 2

    rt.config = Config(cfg, cache_nodes=True)
    c1 = node.instantiate("echo_node")
    rt.config._conf.echo_node.value = 3
    c2 = node.instantiate("echo_node")
    assert c1 is c2


def test_instantiate_resolves_presets(runtime_factory):
    @node.define()
    def add(x: int, y: int) -> int:
        return x + y

    module_name = __name__
    this_module = sys.modules[module_name]
    setattr(this_module, "add", add)

    cfg = {
        "sum_node": {
            "_target_": f"{module_name}.add",
            "x": 2,
            "_use_": "large",
            "_presets_": {
                "small": {"y": 1},
                "large": {"y": 10},
            },
        }
    }
    runtime_factory(config=Config(cfg))

    n = node.instantiate("sum_node")
    assert n() == 12


def test_instantiate_invalid_config(runtime_factory):
    @node.define()
    def noop(x: int) -> int:
        return x

    module_name = __name__
    this_module = sys.modules[module_name]
    setattr(this_module, "noop", noop)

    cfg = {
        "value": 123,
        "bad_target": {"_target_": "not.a.real.path", "x": 1},
        "ok": {"_target_": f"{module_name}.noop", "x": 1},
    }
    runtime_factory(config=Config(cfg))

    with pytest.raises(ConfigurationError):
        node.instantiate("missing")
    with pytest.raises(ConfigurationError):
        node.instantiate("value")
    with pytest.raises(ConfigurationError):
        node.instantiate("bad_target")
