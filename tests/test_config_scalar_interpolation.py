import node
from node import Config


def test_scalar_interpolation(runtime_factory):
    rt = runtime_factory()

    @node.define()
    def show_year(year: int) -> int:
        return year

    cfg = {
        "year": 2023,
        "show_year": {
            "_target_": f"{__name__}.show_year",
            "year": "${year}",
        },
    }
    rt.config = Config(cfg)
    n = show_year()
    assert n() == 2023
