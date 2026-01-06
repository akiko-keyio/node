from node import Config


def test_scalar_interpolation(runtime_factory):
    rt = runtime_factory()

    @rt.define()
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
    node = show_year()
    assert node.get() == 2023
