import yaml
from node.node import Flow, Config, ChainCache, MemoryLRU, DiskJoblib


def test_config_from_yaml(tmp_path):
    conf_yaml = """
add:
  y: 10
"""
    data = yaml.safe_load(conf_yaml)
    flow = Flow(config=Config(data), cache=ChainCache([MemoryLRU(), DiskJoblib(tmp_path)]), log=False)

    @flow.task()
    def add(x, y=1):
        return x + y

    node = add(5)
    assert flow.run(node) == 15
