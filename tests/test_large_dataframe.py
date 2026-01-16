"""Test large DataFrame handling in dimension-broadcasted nodes."""
import numpy as np
import pandas as pd
import pytest
from node import dimension, define, configure, MemoryLRU, reset


@pytest.fixture(autouse=True)
def setup_runtime():
    reset()
    configure(cache=MemoryLRU(), workers=1)


@pytest.mark.integration
def test_large_dataframe_return():
    """Verify large DataFrames are handled correctly in broadcasted nodes.
    
    Regression test for: ValueError when returning large DataFrames from
    dimension-broadcasted nodes due to NumPy 2.x slice assignment behavior.
    """
    @dimension("month")
    def months():
        return pd.date_range("2020-01-01", "2020-01-02", freq="MS").tolist()

    @define()
    def generate_large_df(month: pd.Timestamp) -> pd.DataFrame:
        n_rows = 10000  # Large enough to trigger the bug
        return pd.DataFrame({
            "site": np.random.choice(["A", "B"], n_rows),
            "value": np.random.uniform(0, 1, n_rows),
            "time": month,
        })

    @define(reduce_dims="all")
    def merge_results(data_list: list[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(data_list, ignore_index=True)

    m = months()
    data = generate_large_df(month=m)
    result = merge_results(data_list=data)
    
    output = result()
    assert isinstance(output, pd.DataFrame)
    assert len(output) == 10000  # 1 month * 10000 rows
    assert set(output.columns) == {"site", "value", "time"}
