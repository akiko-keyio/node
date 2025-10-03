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

## Use presets from configuration

Nodes can opt into **presets** that bundle together frequently used argument
combinations. Define presets under the `_presets` key for a task in your YAML
configuration and activate them when calling the node:

```yaml
train:
  learning_rate: 0.01
  _presets:
    fast:
      epochs: 5
      batch_size: 32
    accurate:
      epochs: 50
      batch_size: 16
```

```python
@flow.node()
def train(*, learning_rate: float, epochs: int, batch_size: int) -> None:
    ...

train(presets="fast")  # uses the `fast` overrides from the config
```

Multiple presets can be combined using a sequence (e.g. `presets=("fast",
"accurate")`). Later presets override earlier ones, and any explicit keyword
arguments passed to the node always win over preset values.
