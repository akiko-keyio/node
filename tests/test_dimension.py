import numpy as np
import pytest
from node import Node, dimension, define, configure, MemoryLRU

@pytest.fixture(autouse=True)
def setup_runtime():
    configure(cache=MemoryLRU(), workers=1)

@pytest.mark.unit
def test_dimension_definition():
    @dimension()
    def time_dim(year):
        return [f"{year}-01", f"{year}-02"]

    # Trigger eager execution
    times = time_dim(year=2020)

    # Check Vector Node attributes
    assert isinstance(times, Node)
    assert times.dims == ("time_dim",)
    assert times.coords == {"time_dim": ["2020-01", "2020-02"]}
    assert times._items is not None
    assert isinstance(times._items, np.ndarray)
    assert len(times._items) == 2
    
    # Check items are native values (Optimization: No Wrapper Nodes)
    scalar_0 = times._items[0]
    assert scalar_0 == "2020-01"
    
    # Check Logical Inputs conservation
    assert times.inputs == {"year": 2020} or times._bound_inputs == {"year": 2020}

@pytest.mark.unit
def test_dimension_with_name():
    @dimension(name="time")
    def my_time(start, end):
        return list(range(start, end))

    t = my_time(0, 3)
    assert t.dims == ("time",)
    assert t.coords["time"] == [0, 1, 2]
    assert len(t._items) == 3
    assert t._items[0] == 0

@pytest.mark.unit
def test_vector_node_representation():
    @dimension(name="axis")
    def axis_dim():
        return [1, 2]

    v = axis_dim()
    # Check repr (Logical View with dims comment)
    assert "# dims=(axis:2)" in repr(v)
    assert "axis_dim_0 = axis_dim()" in repr(v)

@pytest.mark.unit
def test_broadcast_1d():
    @dimension(name="time")
    def time_gen():
        return [1, 2]
        
    @define()
    def process(t):
        return t * 10
        
    times = time_gen() # Vector Node
    results = process(t=times) # Should broadcast
    
    assert results.dims == ("time",)
    assert results.coords["time"] == [1, 2]
    assert isinstance(results._items, np.ndarray)
    assert results._items.shape == (2,)
    
    # Check items
    # item 0 input should be times._items[0] (which is a raw value 1)
    item0 = results._items[0]
    assert item0.inputs["t"] == times._items[0]

@pytest.mark.unit
def test_broadcast_2d():
    @dimension(name="time")
    def time_gen():
        return [1, 2]
    
    @dimension(name="model")
    def model_gen():
        return ["A", "B", "C"]
        
    @define()
    def predict(t, m):
        return f"{t}_{m}"
        
    ts = time_gen()
    ms = model_gen()
    
    # Broadcast: (time, model) -> (model, time) sorted alphabetically
    grid = predict(t=ts, m=ms)
    
    assert grid.dims == ("model", "time")
    assert grid.coords["model"] == ["A", "B", "C"]
    assert grid.coords["time"] == [1, 2]
    
    assert grid._items.shape == (3, 2)
    
    item_0_0 = grid._items[0, 0] # model=A, time=1
    assert item_0_0.inputs["m"] == ms._items[0]
    assert item_0_0.inputs["t"] == ts._items[0]

@pytest.mark.unit
def test_alignment_zip():
    @dimension(name="time")
    def time_gen():
        return [1, 2]
        
    @define()
    def step1(t):
        return t
        
    @define()
    def step2(x, t):
        return x + t
        
    times = time_gen()
    s1 = step1(t=times) # Vector(time)
    
    # step2 takes 'x' (Vector time) and 't' (Vector time)
    # Since they share 'time' dimension with identical coords, should align (Zip)
    # NOT expand to (time, time)
    final = step2(x=s1, t=times)
    
    assert final.dims == ("time",)
    assert final._items.shape == (2,) # 1-to-1 match

@pytest.mark.unit
def test_dimension_mismatch():
    @dimension(name="time")
    def time_gen(start):
        return [start, start+1]
        
    t1 = time_gen(0) # [0, 1]
    t2 = time_gen(1) # [1, 2]
    
    @define()
    def combine(a, b):
        return a + b
        
    # t1 and t2 both have dim "time"
    # But coords are different instances (and values)
    # Should raise error
    from node.exceptions import DimensionMismatchError
    with pytest.raises(DimensionMismatchError, match="Dimension Mismatch"):
        combine(a=t1, b=t2)

# Legacy gather tests removed


@pytest.mark.unit
def test_repr_comprehensive():
    # 1. Source Vector Node (from @dimension)
    @dimension(name="time")
    def time_gen():
        return [2020, 2021]

    times = time_gen()
    
    # External Repr (Vector Node)
    # Expected: Script view + comment
    # Note: Variable names are generated as {fn_name}_{idx}
    r_vec = repr(times)
    assert "time_gen_0 = time_gen()" in r_vec
    assert "# dims=(time:2)" in r_vec

    # 2. Computed Vector Node (from broadcasting)
    @define()
    def process(t):
        return t + 1

    result = process(t=times)
    
    # External Repr
    r_res = repr(result)
    # Note: With logical view, only time_gen_0 and process_0 are shown
    # (internal items are not expanded)
    assert "process_0 = process(t=time_gen_0)" in r_res
    assert "# dims=(time:2)" in r_res
    
    # Internal Repr
    # result._items[0] is a Node calling process(t=item0)
    # item0 is 2020. It should be inlined.
    res_item0 = result._items[0]
    r_res0 = repr(res_item0)
    assert "# hash =" in r_res0
    # The magical inlining!
    assert "process_0 = process(t=2020)" in r_res0
