import pandas as pd


def test_dataframe_argument(runtime_factory):
    rt = runtime_factory()

    @rt.define()
    def make_df() -> pd.DataFrame:
        return pd.DataFrame({"a": [1, 2]})

    @rt.define()
    def count_rows(df: pd.DataFrame = make_df()) -> int:
        return len(df)

    node = count_rows()
    assert rt.run(node) == 2
