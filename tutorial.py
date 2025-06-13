"""Node Tutorial
===============

This tutorial shows how to use the Node library to build simple DAGs and how
to inject configuration using YAML to override default parameters when running
a flow.
"""

from node.node import Flow, Config
import yaml

# --- Basic usage -----------------------------------------------------------

flow = Flow()

@flow.task()
def add(x, y):
    return x + y

@flow.task()
def square(z):
    return z * z

if __name__ == "__main__":
    result = flow.run(square(add(2, 3)))
    print(result)  # 25

# --- Configuration injection ----------------------------------------------

# The Flow constructor accepts a Config object that provides default parameters
# for tasks. These defaults can be loaded from a YAML file or string so that
# configuration lives outside your code. Explicit arguments override these
# defaults when ``flow.run`` executes.

yaml_text = """
add:
  y: 5
"""

conf = Config(yaml.safe_load(yaml_text))
flow_cfg = Flow(config=conf)

@flow_cfg.task()
def add_cfg(x, y=1):
    return x + y

if __name__ == "__main__":
    result_cfg = flow_cfg.run(add_cfg(2))  # y from YAML overrides default
    print(result_cfg)  # 7

