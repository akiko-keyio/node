"""Tests for node.sweep function."""

import node
from omegaconf import OmegaConf
from node import Config


def test_sweep_basic():
    """Test basic sweep with proper config setup."""
    node.reset()
    
    # Set up config with function defaults referencing a global value
    config = OmegaConf.create({
        "global_multiplier": 1,
        "compute": {
            "multiplier": "${global_multiplier}",
        }
    })
    node.configure(config=Config(config))
    
    @node.define()
    def compute(x: int, multiplier: int):
        return x * multiplier
    
    # Sweep over global_multiplier
    result = node.sweep(
        compute,
        config={"global_multiplier": [1, 2, 3]},
        x=5,
    ).get()
    
    assert result == [5, 10, 15]


# Note: sweep only re-evaluates the target node by creating new instances with
# updated config defaults. Dependencies passed as arguments (Node objects) are
# created *before* the sweep loop and thus retain their initial config state.
# To sweep a pipeline, either sweep the upstream node or wrap the pipeline in
# a single node if possible.



def test_sweep_dot_path():
    """Test sweep with dot-separated config paths."""
    node.reset()
    
    # Create nested config
    config = OmegaConf.create({
        "model": {
            "lr": 0.01,
        },
        "train": {
            "lr": "${model.lr}",
        }
    })
    node.configure(config=Config(config))
    
    @node.define()
    def train(lr: float):
        return lr * 100
    
    result = node.sweep(
        train,
        config={"model.lr": [0.1, 0.2, 0.3]},
    ).get()
    
    assert result == [10.0, 20.0, 30.0]


def test_sweep_empty_config_error():
    """Test that empty config raises error."""
    node.reset()
    node.configure()
    
    @node.define()
    def dummy(x: int = 1):
        return x
    
    try:
        node.sweep(dummy, config={"x": []}).get()
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "empty" in str(e).lower()


def test_sweep_mismatched_lengths_error():
    """Test that mismatched config lengths raise error."""
    node.reset()
    node.configure()
    
    @node.define()
    def dummy(x: int = 1):
        return x
    
    try:
        node.sweep(dummy, config={"x": [1, 2, 3], "y": [1, 2]}).get()
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "length" in str(e).lower()


def test_sweep_dependency_chain_via_config():
    """Test sweep works for dependency chain if injected via Config.
    
    Scenario: node_a -> node_b -> global_config
    When global_config changes, A is re-instantiated. Since A's dependency B
    is injected via Config (references), B is also re-instantiated with the
    new config value.
    """
    node.reset()
    node.configure()
    
    # 1. Define Nodes locally (binds to current Runtime)
    @node.define()
    def node_b(val: int):
        return val * 10
    
    @node.define()
    def node_a(b: int): 
        return b + 5
        
    # 2. Register nodes to module so _target_ can find them
    import sys
    module_name = __name__
    this_module = sys.modules[module_name]
    
    original_a = getattr(this_module, "node_a", None)
    original_b = getattr(this_module, "node_b", None)
    
    setattr(this_module, "node_b", node_b)
    setattr(this_module, "node_a", node_a)
    
    try:
        # 3. Setup Config using dynamic targets
        config = OmegaConf.create({
            "global_val": 1,
            "node_b": {
                "_target_": f"{module_name}.node_b",
                "val": "${global_val}"
            },
            "node_a": {
                "_target_": f"{module_name}.node_a",
                "b": "${node_b}"
            }
        })
        node.get_runtime().config = Config(config)
        
        # 4. Sweep node_a over global_val
        result = node.sweep(
            node_a,
            config={"global_val": [1, 2, 3]}
        ).get()
        
        # Validation
        assert result == [15, 25, 35]
        
    finally:
        # Cleanup
        if original_a:
            setattr(this_module, "node_a", original_a)
        else:
            delattr(this_module, "node_a")
            
        if original_b:
            setattr(this_module, "node_b", original_b)
        else:
            delattr(this_module, "node_b")
