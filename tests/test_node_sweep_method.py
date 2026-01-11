"""Tests for factory.sweep() method and node.sweep() function."""

import node
from node import Config
from omegaconf import OmegaConf


def test_factory_sweep_basic():
    """Test calling .sweep() on a factory (decorated function)."""
    node.reset()
    
    config = OmegaConf.create({
        "rate": 1,
        "process": {
            "factor": "${rate}"
        }
    })
    node.configure(config=Config(config))
    
    @node.define()
    def process(data, factor):
        return data * factor
    
    # Sweep using factory.sweep()
    results = process.sweep(config={"rate": [1, 2, 3]}, data=10).get()
    
    assert results == [10, 20, 30]


def test_factory_sweep_with_fixed_args():
    """Test .sweep() with both swept config and fixed explicit args."""
    node.reset()
    node.configure(config=Config({
        "global_offset": 0,
        "calc": {
            "offset": "${global_offset}"
        }
    }))
    
    @node.define()
    def calc(val, offset):
        return val + offset
    
    # Sweep config while passing explicit arg
    # val=5 is explicit, offset comes from config
    results = calc.sweep(config={"global_offset": [10, 20]}, val=5).get()
    assert results == [15, 25]


def test_factory_sweep_explicit_overrides_config():
    """Test that explicit args override config defaults."""
    node.reset()
    node.configure(config=Config({
        "global_offset": 0,
        "calc": {
            "offset": "${global_offset}"
        }
    }))
    
    @node.define()
    def calc(val, offset):
        return val + offset
    
    # Explicitly set offset=100, sweeping global_offset has NO EFFECT
    results = calc.sweep(config={"global_offset": [1, 2]}, val=5, offset=100).get()
    
    # Both should use offset=100 (explicit arg always wins)
    assert results == [105, 105]


def test_global_sweep_function():
    """Test using the global node.sweep() function."""
    node.reset()
    node.configure(config=Config({
        "multiplier": 1,
        "compute": {
            "factor": "${multiplier}"
        }
    }))
    
    @node.define()
    def compute(x, factor):
        return x * factor
    
    # Use global sweep function
    results = node.sweep(compute, config={"multiplier": [2, 3, 4]}, x=10).get()
    assert results == [20, 30, 40]


if __name__ == "__main__":
    test_factory_sweep_basic()
    test_factory_sweep_with_fixed_args()
    test_factory_sweep_explicit_overrides_config()
    test_global_sweep_function()
    print("All tests passed!")
