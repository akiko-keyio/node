from datetime import datetime

import pandas as pd
from omegaconf import OmegaConf

from nwm import ZTDNWMGenerator
from pathlib import Path

PROJ_ROOT = Path(__file__).parent.parent.parent
PATH_REPORT=PROJ_ROOT / 'reports'
PATH_STORAGE = PROJ_ROOT / "storage"


from node import Flow, RichReporter, ChainCache, MemoryLRU, DiskJoblib, Config, logger

flow = Flow(default_workers=8,
            continue_on_error=True,
            executor='thread',
            config=Config(PROJ_ROOT / 'configs'/'config.yaml'),
            reporter=RichReporter(show_script_line=False),
            cache=ChainCache([MemoryLRU(), DiskJoblib(root=PATH_STORAGE / 'cache')]))

cfg=flow.config._conf

time_list_small=[datetime(2023, 1, 1, 0),
                 datetime(2023, 7, 31, 12)]
time_list=[ ts   for start, end in [
        (datetime(2023, 1, 1, 0), datetime(2023, 1, 31, 13)),
        (datetime(2023, 7, 1, 0), datetime(2023, 7, 31, 13)),
    ]
    for ts in pd.date_range(start, end, freq="12h")]
# time_list=time_list_small
eda_presets="us_grid10"
era5_presets="us_grid10"
