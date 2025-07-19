# Configuration Guide

This project uses **OmegaConf** to manage default arguments for tasks. You can
load configuration from a YAML file or from a Python mapping. OmegaConf allows
you to reference other keys using `${...}` syntax.

## Load from YAML

```python
from node.node import Flow, Config

flow = Flow(config=Config("config.yaml"))
```

Example `config.yaml`:

```yaml
common:
  value: 10
add:
  y: ${common.value}
```

## Load from mapping

```python
conf = Config({"add": {"y": 5}})
flow = Flow(config=conf)
```

All values returned by `Config.defaults()` are resolved, so references are
expanded automatically.
