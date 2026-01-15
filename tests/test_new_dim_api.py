import pytest
import numpy as np
from node.dimension import dimension, _get_layout_from_sig
from node import MemoryLRU

# Stateless API Tests

def test_signature_parser_stateless():
    def my_func(data, time, site):
        pass
        
    known_dims = {"time", "site"}
    dims, others = _get_layout_from_sig(my_func, known_dims)
    
    assert set(dims) == {"time", "site"}
    assert others == ["data"]
    
    # Test partial knowledge
    known_dims_partial = {"time"}
    dims_p, others_p = _get_layout_from_sig(my_func, known_dims_partial)
    assert dims_p == ["time"]
    assert others_p == ["data", "site"] # 'site' is treated as data because it's unknown

def test_auto_wiring_1d_stateless(runtime_factory):
    import node
    runtime_factory(cache=MemoryLRU(), validate=False)
    
    # Setup
    times = [1, 2, 3]
    
    @node.dimension(name='time')
    def get_times():
        return times
        
    @node.define(reduce_dims="all") # Reduction to scalar
    def analyze_1d(data, time):
        # 'time' should be injected from 'data's context
        return {"sum": np.sum(data), "first_time": time[0]}
        
    t = get_times()
    # t carries dim 'time' with coords [1, 2, 3]
    
    # 1. Interaction: analyze_1d(t)
    # gather_and_wire scans t -> finds 'time'
    # Parser uses known={'time'} -> identifies 'time' param as dim
    # res_node is created above or here?
    # Original:
    res_node = analyze_1d(t)
    res = res_node()

    assert res["sum"] == 6
    assert res["first_time"] == 1


def test_auto_wiring_2d_transpose_stateless(runtime_factory):
    import node
    runtime_factory(cache=MemoryLRU())
    
    @node.dimension()
    def time(): return [1, 2]
    
    @node.dimension()
    def site(): return ["A", "B"]
    
    # Broadcast Production: (time, site)
    @node.define()
    def produce(t, s):
        return f"{t}-{s}"
        
    # Reduction Consumption: Signature (time, site) -> Transpose! (Storage is site, time)
    # Wait, produce order depends on input order? 
    # Standard broadcast sorts dimensions alphabetically. 
    # dims=("site", "time") (s, t alphabetic) -> coords shape (2, 2)
    
    @node.define(reduce_dims="all")
    def reduce_transposed(data, time, site):
        return data.shape, data[0, 0]
        
    t = time()
    s = site()
    grid = produce(t, s) 
    
    # Verify grid layout first
    # sorted dims: site, time
    assert grid.dims == ("site", "time")
    
    # Call reduce
    res_node = reduce_transposed(grid)
    shape, first = res_node()
    
    # Requested (time, site) -> should be transposed to (2, 2)
    assert shape == (2, 2)


def test_consistency_violation(runtime_factory):
    import node
    runtime_factory(cache=MemoryLRU())
    
    # Two dimensions with SAME NAME but DIFFERENT COORDS
    @node.dimension(name="time")
    def time_a(): return [1, 2]
    
    @node.dimension(name="time")
    def time_b(): return [3, 4]
    
    ta = time_a()
    tb = time_b()
    
    @node.define(reduce_dims="all")
    def merge(a, b):
        return 0
        
    # Interaction -> Conflict
    from node.exceptions import DimensionMismatchError
    with pytest.raises(DimensionMismatchError, match="Dimension Mismatch"):
        merge(ta, tb)

def test_missing_dim_error_stateless(runtime_factory):
    import node
    runtime_factory(cache=MemoryLRU())
    
    @node.dimension()
    def time(): return [1]
    
    t = time()
    
    @node.define(reduce_dims="all")
    def need_unknown(data, unknown_dim):
        return 0
        
    # 'unknown_dim' is not in inputs (t only has 'time').
    # So Parser sees it as a data arg.
    # Then Python binding fails because 'unknown_dim' is missing.
    
    with pytest.raises(TypeError): 
         need_unknown(t)()
