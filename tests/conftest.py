# ruff: noqa: E402
import sys
from pathlib import Path

# ensure src is on PYTHONPATH
src_path = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(src_path))

import pytest
from node.node import ChainCache, DiskJoblib, Flow, MemoryLRU


@pytest.fixture
def flow_factory(tmp_path):
    def _make(**kwargs) -> Flow:
        cache = kwargs.pop("cache", ChainCache([MemoryLRU(), DiskJoblib(tmp_path)]))
        kwargs.setdefault("continue_on_error", False)
        kwargs.setdefault("validate", False)
        return Flow(cache=cache, **kwargs)

    return _make
