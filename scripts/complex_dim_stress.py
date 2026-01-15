import node
from node import dimension, define, Node
import numpy as np

# Initialize runtime for global @define usage
node.configure(validate=False, cache=node.MemoryLRU())

# 1. Dimension Definitions
@dimension(name="site")
def site_dim():
    return ["BJ", "SH"]

@dimension(name="time")
def time_dim():
    return ["T1", "T2", "T3"]

@dimension(name="model")
def model_dim():
    return ["M1", "M2"]

# 2. DAG Nodes
@define()
def fetch_obs(site, time):
    return f"obs_{site}_{time}"

@define()
def run_model(data, m, config="default"):
    return f"pred_{data}_{m}_{config}"

# NEW: Reducer Definitions
@define(reduce_dims=["time"])
def reduce_to_site_model(data, time):
    print(f"DEBUG: reduce_to_site_model called. input type: {type(data)}")
    if isinstance(data, (list, tuple)) and len(data) > 0:
         print(f"DEBUG: First item type: {type(data[0])}")
    return list(data)

@define(reduce_dims=["site"])
def reduce_to_model(data, site):
    print(f"DEBUG: reduce_to_model called. input type: {type(data)}")
    return list(data)

@define(reduce_dims=["model"])
def reduce_to_scalar(data, model):
    print(f"DEBUG: reduce_to_scalar called. input type: {type(data)}")
    return list(data)

def run_stress_test():
    print("--- Starting Complex Multi-Dimensional Stress Test (Stateless API) ---")
    
    # --- Generation ---
    sites = site_dim()    # (site: 2)
    times = time_dim()    # (time: 3)
    models = model_dim()  # (model: 2)

    # --- Stage 1: 2D Broadcasting ---
    obs = fetch_obs(site=sites, time=times)
    print(f"Stage 1 (obs) shape: {obs._items.shape}, dims: {obs.dims}")
    
    assert "site" in obs.dims and "time" in obs.dims
    assert obs._items.size == 6

    # --- Stage 2: 3D Broadcasting ---
    pred = run_model(data=obs, m=models, config="v1")
    print(f"Stage 2 (pred) shape: {pred._items.shape}, dims: {pred.dims}")
    
    assert "model" in pred.dims and "site" in pred.dims and "time" in pred.dims
    assert pred._items.size == 12 
    
    # --- Stage 3: Progressive Reduction (New API) ---
    print("Stage 3: Progressive Reduction...")
    
    # A. Reduce 'time' -> Result: (model, site)
    by_site = reduce_to_site_model(pred)
    # Force execution to see prints
    print("Executing reduce_to_site_model...")
    by_site() 

    print(f"  - Reduced 'time': {by_site.dims}, shape: {by_site._items.shape}")
    assert by_site.dims == ("model", "site") 
    assert "time" not in by_site.dims
    assert by_site._items.size == 4
    
    # B. Reduce 'site' -> Result: (model,)
    by_model = reduce_to_model(by_site)
    print("Executing reduce_to_model...")
    by_model()
    
    print(f"  - Reduced 'site': {by_model.dims}, shape: {by_model._items.shape}")
    assert by_model.dims == ("model",)
    assert by_model._items.size == 2
    
    # C. Final Aggregation (Scalar)
    final = reduce_to_scalar(by_model)
    print("Executing reduce_to_scalar...")
    res = final()
    
    print(f"  - Final reduction: {final.dims}, items: {final._items}")
    assert final.dims == ()
    assert final.coords == {}
    
    print(f"  - Execution Result Type: {type(res)}")
    print(f"  - Result Sample: {res}")

    print("\n--- Stress Test Passed Successfully! ---")

if __name__ == "__main__":
    try:
        run_stress_test()
    except Exception as e:
        import traceback
        traceback.print_exc()
        exit(1)
