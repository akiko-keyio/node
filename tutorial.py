"""Node Tutorial
===============

This tutorial shows how to use the Node library to build simple DAGs, how to
inject configuration using YAML, and how cache keys are stored on disk.
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
    node=square(add(square(square(2)), square(square(2))))

    result = flow.run(node)
    print(node, result)


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
def add(x, y=1):
    return x + y

if __name__ == "__main__":
    result_cfg = flow_cfg.run(add(2))  # y from YAML overrides default
    print(result_cfg)  # 7


# --- Cache keys and disk storage -------------------------------------------

# Flow caches results in memory and on disk using ``MemoryLRU`` and
# ``DiskJoblib``.  Each node has a unique ``signature`` derived from its
# expression. ``DiskJoblib`` normally stores each result as ``<md5>.pkl`` using
# the MD5 hash of that signature.  The class uses :class:`FileLock` to guard
# concurrent writes (falling back to a ``_nullcontext`` when locking is
# disabled).

# Passing ``pretty=True`` keeps cache file names partly readable by prefixing a
# sanitized snippet of the signature before the hash.

if __name__ == "__main__":
    from node.node import ChainCache, MemoryLRU, DiskJoblib

    disk = DiskJoblib(".cache", pretty=True)
    flow_cache = Flow(cache=ChainCache([MemoryLRU(), disk]))

    @flow_cache.task()
    def inc(x):
        return x + 1

    n = inc(3)
    print(flow_cache.run(n))

