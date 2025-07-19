import pandas as pd


def test_dataframe_argument(flow_factory):
    flow = flow_factory()

    @flow.node()
    def make_df() -> pd.DataFrame:
        return pd.DataFrame({"a": [1, 2]})

    @flow.node()
    def count_rows(df: pd.DataFrame = make_df()) -> int:
        return len(df)

    node = count_rows()
    assert flow.run(node) == 2
