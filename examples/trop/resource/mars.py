#!/usr/bin/env python
from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import pandas as pd
from loguru import logger
from ecmwfapi import ECMWFService
from ecmwfapi.api import APIRequest

from trop import flow
from trop.resource import PATH_NWM

# presets for EDA download
elda_pressure_level = {
    "class": "od", "expver": 1,
    "levtype": "pl",
    "levelist": [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 850, 900, 925,
                 950, 1000],
    "number": list(range(51)),
    "param": ["129.128", "130.128", "133.128"],
    "stream": "elda", "type": "an",
    "truncation": "NONE", "intgrid": "NONE", "format": "netcdf",
    "anoffset": 9
}
lwda_pressure_level=elda_pressure_level.copy()
lwda_pressure_level.pop('number')
lwda_pressure_level['stream']='lwda'

model_level_boundary = {
    **elda_pressure_level, "levtype": "ml", "levelist": [1, 137]
}

PRESETS = {
    "lwda_global_grid25":lwda_pressure_level | { "grid": "0.25/0.25"},
    "global_raw": elda_pressure_level ,
    "global_grid25": elda_pressure_level | { "grid": "0.25/0.25"},
    "us_grid25": elda_pressure_level | { "grid": "0.25/0.25", "area": "50/-130/20/-60"},
    "us_grid10": elda_pressure_level | { "grid": "0.1/0.1", "area": "50/-130/20/-60"},
    "us_grid25_ml": model_level_boundary | {"grid": "0.25/0.25", "area": "50/-130/20/-60"},
}

# ---------------------------------------------------------------------------
# Patch ECMWFService API so it returns the download URL
# ---------------------------------------------------------------------------
# monkey_patch_ecmwfapi_aria2c.py
import os,sys
import subprocess
from ecmwfapi.api import APIRequest, APIException,robust  # pip install ecmwf-api-client

@robust
def _transfer_with_aria2c(self, url, path, size):
    logger.info(f"Transferring {url} To {path}")
    # 把虚拟环境的 Scripts 目录插到 PATH 最前面
    venv_scripts = os.path.dirname(sys.executable)
    os.environ["PATH"] = venv_scripts + os.pathsep + os.environ.get("PATH", "")
    cmd = [
        "aria2c", "-c",
        "--no-conf",
        "--split=16", "--max-connection-per-server=16",
        "--file-allocation=none",
        "-d", os.path.dirname(path) or ".",
        "-o", os.path.basename(path),
        "--log-level=debug",
        url,
    ]
    try:
        subprocess.check_call(cmd)
    except FileNotFoundError:
        raise APIException("aria2c not found – install it or adjust PATH")
    except subprocess.CalledProcessError as e:
        logger.error(f"aria2c returned {e.returncode}")
        return 0
    return os.path.getsize(path)

APIRequest._transfer = _transfer_with_aria2c

@flow.node(cache=False)
def elda(ts: datetime, preset_name: str) -> Path:
    """download one EDA analysis and return filepath"""

    server = ECMWFService("mars")

    base_out = Path(PATH_NWM/'EDA').expanduser().resolve()
    base_out.mkdir(parents=True, exist_ok=True)
    out_dir = base_out/preset_name
    out_dir.mkdir(exist_ok=True)

    params = PRESETS[preset_name]

    req = params | {"date": ts.strftime("%Y-%m-%d"), "time": ts.strftime("%H:%M:%S")}
    grid = req['grid'].split("/")[0].replace(".", "")
    filename = f"{req['stream']}_{req['levtype']}_{req['anoffset']}_{grid}_{ts:%Y%m%d%H}"
    filepath = out_dir / filename

    if not filepath.exists():
        try:
            raise Exception(f"Disabled Download: {filepath}")
            server.execute(req, str(filepath))
            logger.info(f"got: {filename}")
        except Exception as e:
            logger.warning(f"fail {filename}: {e}")
            raise
    else:
        logger.info(f"Skip {filename} as file already exists")
    return filepath

def main():
    periods = [
        (datetime(2023, 7, 1, 0), datetime(2023, 7, 31, 12)),
        (datetime(2023, 1, 1, 0), datetime(2023, 1, 31, 12)),
    ]
    freq = "12h"
    for preset_name in ["us_grid25", "us_grid10","global_grid25"]:
        for start, end in periods:
            for ts in pd.date_range(start, end, freq=freq):
                elda(ts, preset_name).generate()

def main2():

    periods = [
        (datetime(2023, 1, 1, 0), datetime(2023, 12, 31, 12)),
    ]
    freq = "12h"
    for preset_name in ["global_grid25","lwda_global_grid25"]:
        for start, end in periods:
            for ts in pd.date_range(start, end, freq=freq):
                elda(ts, preset_name).generate()

if __name__ == "__main__":
    main()
    main2()
