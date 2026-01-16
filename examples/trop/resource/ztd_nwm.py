from datetime import datetime

import pandas as pd
from pydantic.v1 import validate_arguments

from trop.reference.location import ngl_station, ngl_station_extend
from trop.resource.cds import era5
from trop.resource.mars import elda
from node import gather, logger
from trop import flow, time_list, time_list_small, cfg
from trop.reference.time import times
from nwm import ZTDNWMGenerator

@flow.node(local=True,cache=False)
def nwm_file(source, preset, time:datetime):
    logger.info(f"Retreiving {source} {preset} {repr(time)}")
    if source =='eda':
        return elda(preset_name=preset, ts=time).get()
    elif source == 'era5':
        return era5(preset=preset, ts=time)
    else:
        raise ValueError(f'Unknown source {source}')

@flow.node(local=True)
def ztd_nwm(nwm_file, location):
    df = ZTDNWMGenerator(nwm_file,location).run()
    if "number" in df.columns:
        df = (df.groupby(["time", "site"])["ztd_simpson"]
              .agg(["mean", "std"])
              .reset_index()
              .rename(columns={"mean": "ztd_nwm", "std": "ztd_nwm_sigma"}))
    else:
        df=df.rename(columns={"ztd_simpson": "ztd_nwm"})

    if df.shape[0] ==0:
        raise Exception("No data found")
    return df

@flow.node(local=True)
def ztd_nwm_origin(nwm_file, location):
    df = ZTDNWMGenerator(nwm_file,location).run()
    return df

@flow.node(local=True)
def ztd_nwm_agg(ztd_nwm_origin,use_member='all'):
    df = ztd_nwm_origin
    if "number" not in df.columns:
        return df.rename(columns={"ztd_simpson": "ztd_nwm"})

    if use_member=='all':
        pass
    elif isinstance(use_member,(int,float)):
        df = df[df.number==use_member]
    elif len(use_member) == 2 \
        and isinstance(use_member[0], int) \
        and isinstance(use_member[1], int):
        if use_member[0] > use_member[1]:
            raise ValueError(f"Invalid member range: {use_member[0]} > {use_member[1]}")
        use_members=range(use_member[0],use_member[1]+1)
        df=df[df.number.isin(use_members)]
    else:
        raise ValueError(f'Unknown use_member {use_member}')

    df = (df.groupby(["time", "site"])["ztd_simpson"]
          .agg(["mean", "std"])
          .reset_index()
          .rename(columns={"mean": "ztd_nwm", "std": "ztd_nwm_sigma"}))

    return df


def ztd_nwms():
    times_list=times().get()
    ztd_nwm_list=[]
    for time in times_list:
        cfg.time=str(time)
        ztd_nwm_list.append(ztd_nwm_origin())
    return gather(ztd_nwm_list)
def ztd_nwm_aggs():
    times_list=times().get()
    ztd_nwm_list=[]
    for time in times_list:
        cfg.time=str(time)
        ztd_nwm_list.append(ztd_nwm_agg())
    return gather(ztd_nwm_list)

def ztd_nwms_extend():
    times_list=times().get()
    ztd_nwm_list=[]
    for time in times_list:
        cfg.time=str(time)
        ztd_nwm_list.append(ztd_nwm(location=ngl_station_extend()))
    return gather(ztd_nwm_list)

@flow.node(local=True)
def ztd_grid_3d(nwm_file, location,resample_h:tuple=(0,6000,100)):
    import numpy as np
    location['alt']=np.nan
    df = ZTDNWMGenerator(nwm_file,location,resample_h=resample_h,interp_to_site=False,vertical_level="h").run()
    return df

def ztd_nwms_grid_3ds():
    times_list=times().get()
    ztd_nwm_list=[]
    for time in times_list:
        cfg.time=str(time)
        ztd_nwm_list.append(ztd_grid_3d())
    return gather(ztd_nwm_list)

if __name__ == "__main__":
    cfg.times=cfg.times_presets.full
    cfg.nwm_file = cfg.nwm_file_presets.eda_global
    print(ztd_nwms())
    ztd_nwms().get()
    # cfg.times=cfg.times_presets.month7
    # ztd_nwms().get()
    #
    # cfg.region='us'
    # cfg.nwm_file=cfg.nwm_file_presets.eda
    # cfg.times=cfg.times_presets.month1
    # ztd_nwms().get()
    # cfg.times=cfg.times_presets.month7
    # ztd_nwms().get()
    #
    # cfg.times=cfg.times_presets.month1
    # ztd_grid_3d().get()
    # cfg.times=cfg.times_presets.month7
    # ztd_grid_3d().get()






