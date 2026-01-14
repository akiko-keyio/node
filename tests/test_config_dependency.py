import node
from node import Config
import sys


def test_config_dependency(tmp_path):
    node.reset()
    node.configure(validate=False)

    @node.define()
    def taska(param1: int, param2: int) -> int:
        return param1 + param2

    @node.define()
    def taskb(depend: int, param1: int) -> int:
        return depend + param1

    module_name = __name__
    this_module = sys.modules[module_name]
    setattr(this_module, "taska", taska)
    setattr(this_module, "taskb", taskb)

    cfg = {
        "taska": {
            "_target_": f"{module_name}.taska",
            "param1": 2,
            "param2": 3,
        },
        "taskb": {
            "depend": "${taska}",
            "param1": 5,
        },
    }
    node.get_runtime().config = Config(cfg)
    task_node = taskb()
    assert task_node() == 10
