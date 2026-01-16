from datetime import datetime

import pandas as pd

from trop import flow


@flow.node()
def times(start:datetime,end:datetime,freq:str):
    return pd.date_range(start=start,end=end,freq=freq).to_pydatetime().tolist()
