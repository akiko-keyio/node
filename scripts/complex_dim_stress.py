import node
from node import dimension, define, gather, Node

# Initialize runtime for global @define usage
node.configure()

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

@define()
def aggregate_site(site_list):
    return f"agg_site({len(site_list)})"

@define()
def final_report(model_results):
    return f"report({len(model_results)})"

def run_stress_test():
    print("--- Starting Complex Multi-Dimensional Stress Test ---")
    
    # --- Generation ---
    sites = site_dim()    # (site: 2)
    times = time_dim()    # (time: 3)
    models = model_dim()  # (model: 2)

    # --- Stage 1: 2D Broadcasting ---
    # obs: site(2) x time(3) -> shape (2, 3)
    # Note: In our implementation, dims are sorted alphabetically by default if multiple new ones appear
    # But here we just check if they are present and shape is correct.
    obs = fetch_obs(site=sites, time=times)
    print(f"Stage 1 (obs) shape: {obs._items.shape}, dims: {obs.dims}")
    
    assert "site" in obs.dims and "time" in obs.dims
    assert obs._items.size == 6
    assert set(obs.coords["site"]) == {"BJ", "SH"}
    assert set(obs.coords["time"]) == {"T1", "T2", "T3"}

    # --- Stage 2: 3D Broadcasting ---
    # pred: obs(site, time) x models(model) -> 3D shape
    pred = run_model(data=obs, m=models, config="v1")
    print(f"Stage 2 (pred) shape: {pred._items.shape}, dims: {pred.dims}")
    
    assert "model" in pred.dims and "site" in pred.dims and "time" in pred.dims
    assert pred._items.size == 12 # 2 * 2 * 3
    
    # Verification of a leaf
    # Find indices for M1, SH, T3
    # Our broadcasting logic uses np.meshgrid indexing
    # We can check specific items via node identity if needed, but here we check structure.
    
    # --- Stage 3: Progressive Reduction ---
    print("Stage 3: Progressive Reduction...")
    
    # A. Reduce 'time' -> Result: (model, site)
    by_site = gather(pred, dim="time")
    print(f"  - Reduced 'time': {by_site.dims}, shape: {by_site._items.shape}")
    assert "time" not in by_site.dims
    assert by_site._items.size == 4 # 2 * 2
    
    # Verify reduction item dependencies
    sample_item = by_site._items.flat[0]
    # Each item in the result of gather is a Node that depends on the items of the previous stage
    # along the reduced axis.
    assert len(sample_item.deps_nodes) == 3 # Gathered 3 time steps

    # B. Reduce 'site' -> Result: (model,)
    by_model = gather(by_site, dim="site")
    print(f"  - Reduced 'site': {by_model.dims}, shape: {by_model._items.shape}")
    assert "site" not in by_model.dims
    assert by_model._items.size == 2
    
    # C. Final Aggregation (Scalar)
    final = gather(by_model, dim="model")
    print(f"  - Final reduction: {final.dims}, items: {final._items}")
    assert final.dims == ()
    assert len(final.deps_nodes) == 2 # Gathered 2 models

    # --- Stage 4: Logical Script Verification ---
    print("Stage 4: Logical Script Verification...")
    script = obs.script
    print("Script Content:")
    print(script)
    
    # It should use the domain/logical names if available
    assert "fetch_obs" in script
    # Our implementation specifically stores _bound_inputs for logical view
    
    print("\n--- Stress Test Passed Successfully! ---")

if __name__ == "__main__":
    try:
        run_stress_test()
    except Exception as e:
        import traceback
        traceback.print_exc()
        exit(1)
