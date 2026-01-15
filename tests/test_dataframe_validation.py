import node
import pandas as pd


def test_dataframe_argument(runtime_factory):
    rt = runtime_factory()

    @node.define()
    def make_df() -> pd.DataFrame:
        return pd.DataFrame({"a": [1, 2]})

    @node.define()
    def count_rows(df: pd.DataFrame = make_df()) -> int:
        return len(df)

    n = count_rows()
    assert rt.run(n) == 2
