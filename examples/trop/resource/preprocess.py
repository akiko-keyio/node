from node import gather, logger

from trop import flow, time_list, time_list_small, cfg
from trop.reference.location import ngl_us, ngl
from trop.resource.cds import era5
from trop.resource.ztd_nwm import ztd_nwm

from trop.utils.outlier_detect import iterative_delete_outliers
from trop.resource.mars import elda



@flow.node(local=True)
def ztd_merge(ztd_nwm,
              ztd_gnss_dataset):
    df_gnss, _, location_gnss,_,_,_ = ztd_gnss_dataset
    ztd = (
        ztd_nwm
        .merge(df_gnss, on=["time", "site"],how="inner")
        .merge(location_gnss[["site", "lon", "lat"]])
    )
    logger.debug(f"merge shape {ztd.shape}")
    if len(ztd)==0:
        raise ValueError
    return ztd

@flow.node()
def ztd_qc(ztd_merge, threshold=5, method='sigma'):
    logger.debug(f"ztd_merge shape in qc {ztd_merge.shape}")
    ztd_merge['res'] = ztd_merge['ztd_nwm'] - ztd_merge['ztd_gnss']
    outlier = iterative_delete_outliers(ztd_merge['res'], threshold=threshold, method=method)
    ztd_qc = ztd_merge[~outlier]
    logger.debug(f"filter out {outlier.sum()} of {len(ztd_qc)}")
    return ztd_qc,outlier.sum()

def ztd_qcs():
    from trop.reference.time import times
    times_list=times().get()
    ztd_qc_list=[]
    for time in times_list:
        cfg.time=str(time)
        ztd_qc_list.append(ztd_qc())
    return gather(ztd_qc_list)


if __name__ == '__main__':
    print(ztd_merge())
    print(ztd_merge().get())
    # ztd_eda_us.get()
    # print(ztd_qcs())
    # print(ztd_qcs().get())