from node.node import Config


def test_scalar_interpolation(flow_factory):
    flow = flow_factory()

    @flow.node()
    def show_year(year: int) -> int:
        return year

    cfg = {
        "year": 2023,
        "show_year": {
            "_target_": f"{__name__}.show_year",
            "year": "${year}",
        },
    }
    flow.config = Config(cfg)
    node = show_year()
    assert node.get() == 2023
