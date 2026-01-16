# era5_node.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import os, sys, subprocess
import pandas as pd
from loguru import logger
from node import Flow, RichReporter, ChainCache, MemoryLRU, DiskJoblib
from pydantic.v1 import validate_arguments

from trop import flow, PATH_STORAGE
from trop.resource import PATH_NWM        # your global data root
from trop.resource.download import transfer_with_retry, transfer_with_idm

import cdsapi
from itertools import cycle

# ──────────────────────────────────────────────────────────────────────────────
# 1. Presets
# ──────────────────────────────────────────────────────────────────────────────
PL_BASE = dict(
    dataset="reanalysis-era5-pressure-levels",
    product_type=["reanalysis"],
    short_name="pl",
    variable=["geopotential", "specific_humidity", "temperature"],
    pressure_level=[  # WMO mandatory + extra
        "1","2","3","5","7","10","20","30","50","70","100","125","150","175",
        "200","225","250","300","350","400","450","500","550","600","650","700",
        "750","775","800","825","850","875","900","925","950","975","1000"
    ],
    data_format="netcdf",
    download_format="unarchived",
)
PL_EN = dict(
    dataset="reanalysis-era5-pressure-levels",
    product_type=["ensemble_mean"],
    short_name="pl",
    variable=["geopotential", "specific_humidity", "temperature"],
    pressure_level=[  # WMO mandatory + extra
        "1","2","3","5","7","10","20","30","50","70","100","125","150","175",
        "200","225","250","300","350","400","450","500","550","600","650","700",
        "750","775","800","825","850","875","900","925","950","975","1000"
    ],
    data_format="netcdf",
    download_format="unarchived",
)


LAND_BASE = dict(
    dataset="reanalysis-era5-land",
    short_name="land",
    variable=[
        "2m_dewpoint_temperature",
        "2m_temperature",
        "surface_pressure",
# "geopotential"
        # "total_precipitation",
    ],
    data_format="netcdf",
    download_format="unarchived",
)

GEO=dict(
    dataset="reanalysis-era5-land",
    short_name="geo",
    variable=[
"geopotential"
    ],
    data_format="netcdf",
    download_format="unarchived",
)
GEOGRID=dict(
    dataset="reanalysis-era5-land",
    short_name="geo_grid",
    variable=[
"geopotential"
    ],
    data_format="grib",
    download_format= "unarchived"
)
SL_BASE = dict(
    dataset="reanalysis-era5-single-levels",
    short_name="sl",
    product_type= ["reanalysis"],
    variable=[
        "2m_dewpoint_temperature",
        "2m_temperature",
        "surface_pressure",
        "geopotential"
        # "total_precipitation",
    ],
    data_format="netcdf",
    download_format="unarchived",
)

HOURLY_PRESETS = {
    "us_pl": PL_BASE | {"area":"50/-130/20/-60"},
    "global_pl": PL_BASE ,
    "global_pl_en":PL_EN,
    "global_land": LAND_BASE,
    "global_land_25":LAND_BASE|{"grid":"0.25/0.25"},
    "global_sl": SL_BASE,
    "geo":GEO,
    "geo_grid":GEOGRID
}

DAILY_PRESETS   = {f"{k}_day": v | {"time":[f"{h:02d}:00" for h in range(24)]} for k, v in HOURLY_PRESETS.items()}
MONTHLY_PRESETS = {f"{k}_mon": v | {"time":[f"{h:02d}:00" for h in range(24)]} for k, v in HOURLY_PRESETS.items()}

PRESETS = {**HOURLY_PRESETS, **DAILY_PRESETS, **MONTHLY_PRESETS}

# ──────────────────────────────────────────────────────────────────────────────
# 2. Key pool for round-robin
# ──────────────────────────────────────────────────────────────────────────────
_keypool = [
    "b8c4e16a-df36-43ee-8577-c445d41a7e49", #apm
    # "2a52b286-81e8-4f23-8ad0-4f17ad1b5d7e" #txz
    # "efe77d01-5f17-4278-880c-b44df1088ce0", #cgx
]
_key_cycle = cycle(_keypool)

# ──────────────────────────────────────────────────────────────────────────────
# 3. The flow node
# ──────────────────────────────────────────────────────────────────────────────
def era5(ts: datetime, preset: str) -> Path:
    """Download ERA5 NetCDF for given timestamp & preset (hour/day/month)."""
    logger.debug(f"{ts}-{preset}")
    out_root = Path(PATH_NWM, "ERA5").expanduser()
    out_dir  = out_root / preset
    out_dir.mkdir(parents=True, exist_ok=True)

    req   = PRESETS[preset].copy()
    name  = req.pop("dataset")
    short_name=req.pop("short_name")
    req["year"], req["month"] = f"{ts.year:04d}", f"{ts.month:02d}"

    if preset.endswith("_day"):
        req["day"] = f"{ts.day:02d}"
        tag_time   = ts.strftime("%Y%m%d")
    elif preset.endswith("_mon"):
        req["day"] = [f"{d:02d}" for d in range(1,32)]
        tag_time   = ts.strftime("%Y%m")
    else:
        req["day"], req["time"] = f"{ts.day:02d}", f"{ts.hour:02d}:00"
        tag_time   = ts.strftime("%Y%m%d%H")

    grid_tag = req.get("grid","native").split("/")[0].replace(".","")
    fname    = f"era5_{short_name}_{grid_tag}_{tag_time}.nc"
    fpath    = out_dir / fname

    if fpath.exists():
        logger.info(f"skip {fname}")
        return fpath

    raise Exception(f"disabled download {ts} {preset}")

    api_key = next(_key_cycle)
    logger.info(f"Requesting ERA5 {preset} @ {fname} using key {api_key}")
    logger.debug(req)

    client = cdsapi.Client(url="https://cds.climate.copernicus.eu/api", key=api_key)
    result = client.retrieve(name, req)
    url    = result.location
    logger.info(f"downloaded {url} to {fpath} with {api_key}")

    transfer_with_idm(url, fpath)
    return fpath

# ──────────────────────────────────────────────────────────────────────────────
# 4. Batch helper
# ──────────────────────────────────────────────────────────────────────────────
def hourly_loop(start: datetime, end: datetime, preset="global_pl"):
    """one file per hour"""
    from node import gather
    gather([
        era5(ts, preset)
        for ts in pd.date_range(start, end, freq="12H")
    ]).get()

# ──────────────────────────────────────────────────────────────────────────────
# 5. 示例
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    era5(datetime(2023, 1, 1), "geo").get()
    era5(datetime(2023, 1, 1), "geo_grid").get()
    for preset in ["global_pl_en","global_sl","global_land_25","global_pl"]:
        hourly_loop(datetime(2023, 1, 1), datetime(2023, 1, 3),preset=preset)
        hourly_loop(datetime(2023, 7, 1), datetime(2023, 7, 3), preset=preset)
    for preset in ["global_pl"]:
        hourly_loop(datetime(2023, 1, 1), datetime(2023, 12, 31),preset=preset)
        # hourly_loop(datetime(2023, 7, 1), datetime(2023, 7, 31),preset=preset)

