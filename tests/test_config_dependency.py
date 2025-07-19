from node.node import Flow, Config

flow = Flow(validate=False)


@flow.node()
def taska(param1: int, param2: int) -> int:
    return param1 + param2


@flow.node()
def taskb(depend: int, param1: int) -> int:
    return depend + param1


def test_config_dependency(tmp_path):
    cfg = {
        "taska": {
            "_target_": "test_config_dependency.taska",
            "param1": 2,
            "param2": 3,
        },
        "taskb": {
            "depend": "${taska}",
            "param1": 5,
        },
    }
    config = Config(cfg)
    flow.config = config
    node = taskb()
    assert node.get() == 10
