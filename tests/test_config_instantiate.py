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


def test_instantiate_sweep_top_level(runtime_factory):
    @node.define()
    def score(degree: int, basis: str) -> str:
        return f"{basis}:{degree}"

    module_name = __name__
    this_module = sys.modules[module_name]
    setattr(this_module, "score", score)

    cfg = {
        "eval_node": {
            "_target_": f"{module_name}.score",
            "degree": 1,
            "basis": "ahsh",
        }
    }
    runtime_factory(config=Config(cfg))

    n = node.instantiate(
        "eval_node",
        sweep={
            "basis": ["ahsh", "poly"],
            "degree": [1, 2, 3],
        },
    )
    result = n()
    assert result.dims == ("sweep_basis", "sweep_degree")
    assert result.shape == (2, 3)
    assert result[1, 2] == "poly:3"


def test_instantiate_sweep_dotted_path_with_existing_dimension(runtime_factory):
    @node.dimension(name="time")
    def time_dim():
        return [10, 20]

    @node.define()
    def load(t: int) -> int:
        return t

    @node.define()
    def evaluate(x: int, trop_ls: dict[str, object]) -> str:
        return f"{trop_ls['basis']}:{trop_ls['degree']}:{x}"

    module_name = __name__
    this_module = sys.modules[module_name]
    setattr(this_module, "time_dim", time_dim)
    setattr(this_module, "load", load)
    setattr(this_module, "evaluate", evaluate)

    cfg = {
        "time_node": {
            "_target_": f"{module_name}.time_dim",
        },
        "load_node": {
            "_target_": f"{module_name}.load",
            "t": "${time_node}",
        },
        "eval_node": {
            "_target_": f"{module_name}.evaluate",
            "x": "${load_node}",
            "trop_ls": {
                "basis": "ahsh",
                "degree": 1,
            },
        },
    }
    runtime_factory(config=Config(cfg))

    n = node.instantiate(
        "eval_node",
        sweep={
            "trop_ls.basis": ["ahsh", "poly"],
            "trop_ls.degree": [1, 2],
        },
    )
    result = n()

    assert result.dims == ("sweep_trop_ls_basis", "sweep_trop_ls_degree", "time")
    assert result.shape == (2, 2, 2)
    assert result[1, 1, 0] == "poly:2:10"


def test_instantiate_sweep_supports_referenced_subnode_config(runtime_factory):
    @node.dimension(name="dim_hours")
    def dim_hours():
        return [1, 2]

    @node.define()
    def trop_ls(trop_matrix: int, grid_location: str, basis: str, degree: int) -> str:
        return f"{basis}:{degree}:{trop_matrix}:{grid_location}"

    @node.define()
    def trop_eval(trop_ls: str, dim_hours: int) -> str:
        return f"{trop_ls}|h={dim_hours}"

    module_name = __name__
    this_module = sys.modules[module_name]
    setattr(this_module, "dim_hours", dim_hours)
    setattr(this_module, "trop_ls", trop_ls)
    setattr(this_module, "trop_eval", trop_eval)

    cfg = {
        "dim_hours": {"_target_": f"{module_name}.dim_hours"},
        "trop_matrix": 7,
        "grid_location": "G0",
        "trop_ls": {
            "_target_": f"{module_name}.trop_ls",
            "trop_matrix": "${trop_matrix}",
            "grid_location": "${grid_location}",
            "basis": "ahsh",
            "degree": 15,
        },
        "trop_eval": {
            "_target_": f"{module_name}.trop_eval",
            "trop_ls": "${trop_ls}",
            "dim_hours": "${dim_hours}",
        },
    }
    runtime_factory(config=Config(cfg))

    n = node.instantiate(
        "trop_eval",
        sweep={
            "trop_ls.basis": ["ahsh", "poly"],
            "trop_ls.degree": [1, 2, 3],
        },
    )
    result = n()
    assert set(result.dims) == {"sweep_basis", "sweep_degree", "dim_hours"}
    assert result.coords["sweep_basis"] == ["ahsh", "poly"]
    assert result.coords["sweep_degree"] == [1, 2, 3]
    assert result.coords["dim_hours"] == [1, 2]
    shape_by_dim = dict(zip(result.dims, result.shape, strict=True))
    assert shape_by_dim == {
        "sweep_basis": 2,
        "sweep_degree": 3,
        "dim_hours": 2,
    }
    idx_basis = result.coords["sweep_basis"].index("poly")
    idx_degree = result.coords["sweep_degree"].index(3)
    idx_hour = result.coords["dim_hours"].index(1)
    index = tuple(
        {
            "sweep_basis": idx_basis,
            "sweep_degree": idx_degree,
            "dim_hours": idx_hour,
        }[dim]
        for dim in result.dims
    )
    assert result[index] == "poly:3:7:G0|h=1"
