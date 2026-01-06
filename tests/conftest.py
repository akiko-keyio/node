# ruff: noqa: E402
import sys
from pathlib import Path

# ensure src is on PYTHONPATH
src_path = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(src_path))

import pytest
from node import Runtime, ChainCache, DiskJoblib, MemoryLRU, reset


@pytest.fixture(autouse=True)
def reset_runtime():
    """Reset global runtime before each test."""
    reset()
    yield
    reset()


@pytest.fixture
def runtime_factory(tmp_path):
    def _make(**kwargs) -> Runtime:
        cache = kwargs.pop("cache", ChainCache([MemoryLRU(), DiskJoblib(tmp_path)]))
        kwargs.setdefault("continue_on_error", False)
        kwargs.setdefault("validate", False)
        return Runtime(cache=cache, **kwargs)

    return _make
