import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from nwm import ZTDNWMGenerator
from sklearn.neighbors import BallTree
from trop.ml.ml import *

from trop import flow
from trop.resource.ztd_nwm import nwm_file


def add_knn_density(df: pd.DataFrame,
                    lat_col: str = "lat",
                    lon_col: str = "lon",
                    site_col: str = "site",
                    k: int = 3,
                    earth_radius_m: float = 6371000.0,
                    min_sep_m: float = 1.0,
                    extra_neighbors: int = 20,
                    co_loc_thresh_km: float = 0.01) -> pd.DataFrame:
    """
    计算 k 近邻密度与有效间距（球面大圆距离）。
    额外规则：若存在同址点（与任一其他站 < co_loc_thresh_km），则将 eff_spacing_km 置 0。

    输出列：
      rk_km                第 k 个非零近邻距离 (km)
      knn_mean_dist_km     前 k 个非零近邻平均距离 (km)
      lambda_k_per_km2     kNN 密度估计 k/(π*rk^2) (站/km²)
      eff_spacing_km       有效间距 rk*sqrt(π/k) (km)，同址则强制 0
    """
    out = df.copy()
    valid = out[lat_col].notna() & out[lon_col].notna()
    if valid.sum() == 0:
        return out

    coords_rad = np.radians(out.loc[valid, [lat_col, lon_col]].to_numpy())
    tree = BallTree(coords_rad, metric="haversine")

    # --- kNN（去自身与“零距离”邻点）
    k_query = min(len(coords_rad), k + 1 + extra_neighbors)
    dist_rad, ind = tree.query(coords_rad, k=k_query)

    min_sep_rad = (min_sep_m / earth_radius_m)          # 1 m 缺省
    rk_km = np.full(len(coords_rad), np.nan, dtype=float)
    mean_k_km = np.full(len(coords_rad), np.nan, dtype=float)

    for i in range(len(coords_rad)):
        d = dist_rad[i, :]
        mask_nz = d > min_sep_rad                       # 非零/非同址邻点
        d_k = d[mask_nz][:k]
        if d_k.size == k:
            d_k_km = (d_k * earth_radius_m) / 1000.0
            rk_km[i] = d_k_km[-1]
            mean_k_km[i] = d_k_km.mean()

    out.loc[valid, "rk_km"] = rk_km
    out.loc[valid, "knn_mean_dist_km"] = mean_k_km

    rk = out.loc[valid, "rk_km"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        lambda_k = k / (np.pi * np.maximum(rk, 1e-9)**2)   # 站/km²
        leff = rk * np.sqrt(np.pi / k)                     # km

    out.loc[valid, "lambda_k_per_km2"] = lambda_k
    out.loc[valid, "eff_spacing_km"] = leff

    # --- 同址覆盖：若 < co_loc_thresh_km 内存在其他站，则 eff_spacing_km = 0
    eps_rad = (co_loc_thresh_km * 1000.0) / earth_radius_m
    # query_radius 返回包含自身的索引列表；长度>1 表示半径内有其他站
    within_eps = tree.query_radius(coords_rad, r=eps_rad, return_distance=False)
    has_coloc = np.array([len(ix) > 1 for ix in within_eps])

    valid_idx = out.index[valid].to_numpy()
    out.loc[valid_idx[has_coloc], "eff_spacing_km"] = 0.0

    return out

def add_nearest_gc_distance(df: pd.DataFrame,
                            lat_col: str = "lat",
                            lon_col: str = "lon",
                            site_col: str = "site",
                            earth_radius_m: float = 6371000.0) -> pd.DataFrame:
    """
    为 df 中每个点找到球面（大圆）最近邻，并新增两列：
    - 'nearest_site' 最近站点名
    - 'nearest_dist_km' 最近球面距离（km）
    需要 sklearn；若没有可 pip 安装 scikit-learn。
    """
    # 只在经纬度非空的行上计算；其余保持 NaN
    valid = df[lat_col].notna() & df[lon_col].notna()
    coords_rad = np.radians(df.loc[valid, [lat_col, lon_col]].to_numpy())

    # BallTree + haversine（返回的是弧度距离）
    from sklearn.neighbors import BallTree
    tree = BallTree(coords_rad, metric="haversine")
    dist_rad, ind = tree.query(coords_rad, k=2)   # 第0个是自身，取第1个

    # 最近邻索引（映射回原 df 的索引）
    valid_idx = df.index[valid].to_numpy()
    nn_idx = valid_idx[ind[:, 1]]

    # 距离（米 → 千米）
    nn_dist_km = (dist_rad[:, 1] * earth_radius_m) / 1000.0

    # 写回副本
    out = df.copy()
    out.loc[valid_idx, "nearest_site"] = df.loc[nn_idx, site_col].to_numpy()
    out.loc[valid_idx, "nearest_dist_km"] = nn_dist_km

    return out


def add_nearest_gc_distance(df: pd.DataFrame,
                            lat_col: str = "lat",
                            lon_col: str = "lon",
                            site_col: str = "site",
                            earth_radius_m: float = 6371000.0) -> pd.DataFrame:
    """
    为 df 中每个点找到球面（大圆）最近邻，并新增两列：
    - 'nearest_site' 最近站点名
    - 'nearest_dist_km' 最近球面距离（km）
    需要 sklearn；若没有可 pip 安装 scikit-learn。
    """
    # 只在经纬度非空的行上计算；其余保持 NaN
    valid = df[lat_col].notna() & df[lon_col].notna()
    coords_rad = np.radians(df.loc[valid, [lat_col, lon_col]].to_numpy())

    # BallTree + haversine（返回的是弧度距离）
    from sklearn.neighbors import BallTree
    tree = BallTree(coords_rad, metric="haversine")
    dist_rad, ind = tree.query(coords_rad, k=2)   # 第0个是自身，取第1个

    # 最近邻索引（映射回原 df 的索引）
    valid_idx = df.index[valid].to_numpy()
    nn_idx = valid_idx[ind[:, 1]]

    # 距离（米 → 千米）
    nn_dist_km = (dist_rad[:, 1] * earth_radius_m) / 1000.0

    # 写回副本
    out = df.copy()
    out.loc[valid_idx, "nearest_site"] = df.loc[nn_idx, site_col].to_numpy()
    out.loc[valid_idx, "nearest_dist_km"] = nn_dist_km

    return out

# 用法
@flow.node()
def location_info(ztd_gnss_dataset):
    df_filtered, filter_info, location, good_sites_list, sparse_site_lists, good_times_list = ztd_gnss_dataset
    location.region = location.region.fillna("other")
    location.region = location.region.apply(lambda x: x.capitalize() if x == "other" else x.upper())
    location = add_knn_density(location, k=10, co_loc_thresh_km=0.01)
    location = add_nearest_gc_distance(location)
    filter_info['good_sites_count'] = len(good_sites_list)
    filter_info['sparse_sites_count'] = len(sparse_site_lists)
    z = ZTDNWMGenerator(nwm_file().get(), location)
    df = z.check_site_vertical_status()
    hd = df.groupby("site")[['height_diff']].mean().reset_index()
    location=location.merge(hd)
    return location,filter_info

def dups(location):
    dups_mask = location.duplicated(subset=["lon", "lat"], keep=False)
    dups = location[dups_mask].sort_values(["lon", "lat"])

    # 重复坐标
    dups_summary = (
        dups.groupby(["lon", "lat"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    return dups_summary

@flow.node()
def mlresult(location_info,
             test_size=0.0,
             use_member="all",
             regions=['all'],
             times=['full'],
             datasets=['eda_global']):
    location=location_info[0]
    ztd_ml_test_list = []
    ztd_ml_train_list = []
    cfg.ztd_nwm_agg.use_member = use_member
    # cfg.ztd_ngl_dataset=cfg.ztd_ngl_dataset_presets.sparse
    cfg.ml.test_size=test_size

    def val_prepare(df):
        df = df.merge(location[['site', 'alt', 'region', 'nearest_site', 'nearest_dist_km','height_diff']], how="left", on="site")
        df['res'] = df['ztd_nwm'] - df['ztd_gnss']
        df['ztd_nwm_cor'] = df['ztd_nwm'] - df['pred']
        df['res_cor'] = df['res'] - df['pred']
        if False:
            from ztd_hc import HeightCorrection
            df['res'] = HeightCorrection().correct_long_dataframe(df, value_col='res')
            df['res_cor'] = HeightCorrection().correct_long_dataframe(df, value_col='res_cor')
            df['ztd_gnss_sigma'] = HeightCorrection().correct_long_dataframe(df, value_col='ztd_gnss_sigma')
            if 'ztd_nwm_sigma' in df.columns:
                df['ztd_nwm_sigma'] = HeightCorrection().correct_long_dataframe(df, value_col='ztd_nwm_sigma')
        return df

    for region in regions:
        for time_preset in times:
            for dataset in datasets:  # ,'lwda','era5']:
                ms = mls(region=region, nwm_file_preset=dataset, time_preset=time_preset).get()
                if dataset == 'eda_global':
                    dataset_name = 'elda'
                else:
                    dataset_name = dataset

                if flow.config._conf.ml.test_size > 0:
                    ztd_ml_test = pd.concat([m[1] for m in ms])
                    ztd_ml_test = val_prepare(ztd_ml_test)
                    ztd_ml_test['sample'] = 'test'
                    ztd_ml_test['dataset'] = dataset_name
                    ztd_ml_test_list.append(ztd_ml_test)

                ztd_ml_train = pd.concat([m[0] for m in ms])
                ztd_ml_train = val_prepare(ztd_ml_train)
                ztd_ml_train['sample'] = 'train'
                ztd_ml_train['dataset'] = dataset_name
                ztd_ml_train_list.append(ztd_ml_train)

    def prepare(df_list):
        df = pd.concat(df_list)

        # Map Season
        df["month"] = pd.to_datetime(df["time"]).dt.month
        # season_map_north = {
        #     12: "Winter", 1: "Winter", 2: "Winter",
        #     3: "Spring", 4: "Spring", 5: "Spring",
        #     6: "Summer", 7: "Summer", 8: "Summer",
        #     9: "Autumn", 10: "Autumn", 11: "Autumn"
        # }
        season_map = {
            12: 'DJF', 1: 'DJF', 2: 'DJF',
            3: 'MAM', 4: 'MAM', 5: 'MAM',
            6: 'JJA', 7: 'JJA', 8: 'JJA',
            9: 'SON', 10: 'SON', 11: 'SON'
        }
        df["season"] = df["month"].map(season_map)
        # season_map_south = {
        #     12: "Summer", 1: "Summer", 2: "Summer",
        #     3: "Autumn", 4: "Autumn", 5: "Autumn",
        #     6: "Winter", 7: "Winter", 8: "Winter",
        #     9: "Spring", 10: "Spring", 11: "Spring"
        # }
        # south = df["lat"] < 0
        # df.loc[south, "season"] = df.loc[south, "month"].map(season_map_south)

        df['dataset'] = df['dataset'].str.upper()
        df['Season'] = df['season']


        lat = df['lat']
        conds = [
            (lat >= -20) & (lat <= 20),
            (lat > 20),
            (lat < -20)
        ]
        choices = ['Tropics (20°N~20°S)', 'NHE (20°N~90°N)', 'SHE (20°S~90°S)']
        df['LatBand'] = np.select(conds, choices, default='Other')

        return df

    train_site = ztd_ml_train.site.drop_duplicates().values
    if flow.config._conf.ml.test_size > 0:
        df_all = pd.concat([prepare(ztd_ml_train_list), prepare(ztd_ml_test_list)])
    else:
        df_all = prepare(ztd_ml_train_list)

    try:
        df_all['ztd_sigma'] = np.sqrt(df_all['ztd_nwm_sigma'] ** 2 + df_all['ztd_gnss_sigma'] ** 2)
    except:
        df_all['ztd_sigma'] = np.nan
    return df_all,train_site