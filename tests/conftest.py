# ruff: noqa: E402
import sys
from pathlib import Path

# ensure src is on PYTHONPATH
src_path = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(src_path))

import pytest
from node import configure, get_runtime, reset, ChainCache, DiskJoblib, MemoryLRU


@pytest.fixture(autouse=True)
def reset_runtime():
    """Reset global runtime before each test."""
    reset()
    yield
    reset()


@pytest.fixture
def runtime_factory(tmp_path):
    """Factory that configures the global runtime and returns it.
    
    This ensures all tests use the singleton global runtime.
    """
    def _make(**kwargs):
        cache = kwargs.pop("cache", ChainCache([MemoryLRU(), DiskJoblib(tmp_path)]))
        kwargs.setdefault("continue_on_error", False)
        kwargs.setdefault("validate", False)
        # kwargs.setdefault("reporter", None) # This triggers default RichReporter!
        rt = configure(cache=cache, **kwargs)
        if "reporter" not in kwargs:
            rt.reporter = None
        return rt

    return _make
