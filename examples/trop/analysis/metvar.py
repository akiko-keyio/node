#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ERA5 6-hourly single-level time-series sampler for arbitrary sites & times
==========================================================================

功能：
  - 复用 / 下载 ERA5 单层 6 小时、0.25° 的若干单层变量（可自定义，比如 msl / t2m / tcwv）；
  - 在给定的 time 序列上做“最近时刻”匹配（nearest in time）；
  - 在给定的站点经纬度上做空间插值（nearest in space）；
  - 输出一个站点 × 时间的 DataFrame，包含：
        time, site, (以及你指定的气象变量 msl, t2m, tcwv, ...)

假设：
  - 所有时间均为 UTC（如果传入带时区，会自动转为 UTC）。
  - ERA5 数据按月存放在 outdir/raw/single_levels/ 目录下，
    文件名可以是任意 .nc，只要是真正的 NetCDF 即可。
  - 如 download=True，则缺失的月份会通过 cdsapi 自动下载。
"""

import os
import zipfile
import tarfile
import shutil
from typing import Sequence

import numpy as np
import pandas as pd

from trop import PATH_STORAGE, flow

# ---------------- ERA5 配置 ----------------
TIMES_6H = ["00:00", "06:00", "12:00", "18:00"]

# 标准变量名 -> ERA5 请求里的变量名
VAR_NAME_MAP = {
    "msl":    "mean_sea_level_pressure",
    "t2m":    "2m_temperature",
    "tcwv":   "total_column_water_vapour",
    "precip": "total_precipitation",                 # 新增：总降水
    "cape":   "convective_available_potential_energy" # 新增：CAPE
}


# 默认：标准变量名和 ERA5 变量名
DEFAULT_STD_VARS = tuple(VAR_NAME_MAP.keys())      # ("msl", "t2m", "tcwv")
DEFAULT_ERA5_VARS = tuple(VAR_NAME_MAP.values())   # ("mean_sea_level_pressure", ...)

# ---------------- 通用工具函数 ----------------
def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def try_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def looks_like_netcdf(path: str) -> bool:
    """
    判断文件是否为真正的 NetCDF/HDF5。
    个别 CDS 返回的“伪 .nc”其实是 zip/tar。
    """
    try:
        with open(path, "rb") as f:
            hdr = f.read(8)
        return hdr.startswith(b"CDF") or hdr.startswith(b"\x89HDF\r\n\x1a\n")
    except Exception:
        return False


def unpack_nc_archive_if_needed(path: str, outdir: str) -> list:
    """
    若 path 是“伪 .nc 压缩包”，就地解出其中的 .nc 文件到 outdir。
    - 目的：CDS 偶尔会把多文件打包成 zip/tar 并伪装成 .nc，此处自动兜底。
    """
    base = os.path.basename(path)
    if looks_like_netcdf(path):
        return []

    produced = []

    def emit(member_name, src_fp):
        out_path = os.path.join(outdir, os.path.basename(member_name))
        if not os.path.exists(out_path):
            with open(out_path, "wb") as wf:
                shutil.copyfileobj(src_fp, wf)
        produced.append(out_path)

    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as z:
                for nm in z.namelist():
                    if nm.endswith(".nc"):
                        with z.open(nm) as fp:
                            emit(nm, fp)
            print(f"[UNPACK] {base} -> {len(produced)} files (zip)")
            return produced
    except Exception as e:
        print("[UNPACK] ZIP failed:", e)

    try:
        if tarfile.is_tarfile(path):
            with tarfile.open(path) as t:
                for m in t.getmembers():
                    if m.name.endswith(".nc"):
                        fp = t.extractfile(m)
                        if fp is not None:
                            emit(m.name, fp)
            print(f"[UNPACK] {base} -> {len(produced)} files (tar)")
            return produced
    except Exception as e:
        print("[UNPACK] TAR failed:", e)

    print(f"[WARN] {base} is neither NetCDF nor archive; skip.")
    return []


def ensure_unpacked_in_folder(folder: str):
    """
    对目录中所有 .nc 尝试自动解包（若为伪 .nc），确保可被 xarray 读取。
    """
    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith(".nc"):
            continue
        path = os.path.join(folder, fn)
        if not looks_like_netcdf(path):
            unpack_nc_archive_if_needed(path, folder)


# ---------------- ERA5 下载 ----------------
def cds_retrieve_single_level(
    year: int,
    month: int,
    out_nc: str,
    variables=None,
    times_6h=None,
):
    """
    通过 CDS 下载 ERA5 单层 6h, 0.25° 数据（变量可配置）。

    参数
    ----
    year, month : int
        年、月。
    out_nc : str
        输出 NetCDF 文件路径。
    variables : Sequence[str] or None
        传给 CDS 的变量名字（ERA5 官方名字，如 "2m_temperature"）。
        若为 None，则使用 DEFAULT_ERA5_VARS。
    times_6h : Sequence[str] or None
        需要的时间步（例如 TIMES_6H）。若为 None，则用 TIMES_6H。
    """
    import cdsapi

    if variables is None:
        variables = DEFAULT_ERA5_VARS
    if times_6h is None:
        times_6h = TIMES_6H

    c = cdsapi.Client()
    req = {
        "product_type": "reanalysis",
        "variable": list(variables),
        "year": str(year),
        "month": f"{month:02d}",
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": list(times_6h),
        "format": "netcdf",
        "grid": [0.25, 0.25],
    }
    print(f"[CDS] Single-level {year}-{month:02d} -> {out_nc} (vars={variables})")
    c.retrieve("reanalysis-era5-single-levels", req, out_nc)


# ---------------- xarray 辅助函数 ----------------
def harmonize_dims(ds):
    """
    维度统一：
    - valid_time -> time；
    - expver 维取第一个。
    """
    if "valid_time" in ds.dims and "time" not in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    if "expver" in ds.dims:
        ds = ds.isel(expver=0)
    return ds


def get_any(ds, names):
    """从 names 列表中返回第一个存在的变量（xarray.DataArray），否则 None。"""
    for n in names:
        if n in ds:
            return ds[n]
    return None


def pick_latlon_names(da):
    """在变量的维度/坐标中推断经纬度名称，兼容 latitude/longitude 与 lat/lon。"""
    for lat in ("latitude", "lat"):
        if lat in da.dims or lat in da.coords:
            lat_name = lat
            break
    else:
        return None, None

    for lon in ("longitude", "lon"):
        if lon in da.dims or lon in da.coords:
            lon_name = lon
            break
    else:
        return None, None

    return lat_name, lon_name


def detect_and_attach_latlon(ds):
    """
    确保 dataset 级别有 lat/lon 坐标；
    若只存在于某个变量的 coords 中，则提升为 ds.coords。
    """
    for lat_name, lon_name in (("latitude", "longitude"), ("lat", "lon")):
        if lat_name in ds.coords and lon_name in ds.coords:
            return ds, lat_name, lon_name

    for v in ds.data_vars:
        var = ds[v]
        lat_name, lon_name = pick_latlon_names(var)
        if lat_name and lon_name:
            ds = ds.assign_coords(
                {
                    lat_name: var.coords.get(lat_name, var[lat_name]),
                    lon_name: var.coords.get(lon_name, var[lon_name]),
                }
            )
            return ds, lat_name, lon_name

    raise ValueError("无法在数据集中识别经纬度坐标名。")


def build_core_dataset(ds):
    """
    从原始 ERA5 Dataset 中挑出核心变量，并统一变量名为标准名（msl/t2m/tcwv/...）。

    - 标准变量名来自 VAR_NAME_MAP 的 key；
    - 会尝试识别 ERA5 原名（value）或已经是标准名的变量。
    """
    import xarray as xr

    data_vars = {}

    for std_name, era_name in VAR_NAME_MAP.items():
        # 兼容 ERA5 原名和已经重命名过的标准名
        var = get_any(ds, [era_name, std_name])
        if var is not None:
            data_vars[std_name] = var

    if not data_vars:
        raise SystemExit("[ERROR] 输入 NetCDF 未包含 VAR_NAME_MAP 中任何一个变量。")

    return xr.Dataset(data_vars)


def sample_timeseries_at_sites(ds, sites_df):
    """
    在 Dataset(ds: time, lat, lon) 上，对所有 time × sites 做 nearest 插值。

    返回：
        sampled_ds: 维度 (time, site)，包含各气象变量
        lat_name, lon_name: 原 ERA5 的经纬度坐标名（例如 latitude, longitude）
    """
    import numpy as np

    ds, lat_name, lon_name = detect_and_attach_latlon(ds)

    # 从站点表中取经纬度
    if "lon" not in sites_df.columns or "lat" not in sites_df.columns:
        raise SystemExit("sites_df 必须至少包含 lon, lat 列。")
    if "site" not in sites_df.columns:
        sites_df = sites_df.copy()
        sites_df["site"] = np.arange(len(sites_df))

    lons = sites_df["lon"].to_numpy().copy()
    lats = sites_df["lat"].to_numpy().copy()

    # 若 ERA5 经度为 [0,360)，而站点为 [-180,180]，则对站点经度加 360 对齐。
    if float(ds[lon_name].min()) >= 0 and lons.min() < 0:
        lons = (lons + 360.0) % 360.0

    site_ids = sites_df["site"].to_numpy()

    # 使用 "site" 作为新的空间维度
    sampled = ds.interp(
        {
            lat_name: ("site", lats),
            lon_name: ("site", lons),
        },
        method="nearest",
    )
    sampled = sampled.assign_coords(site=("site", site_ids))

    return sampled, lat_name, lon_name


# ---------------- 主函数：对外接口 ----------------
def extract_era5_6h_timeseries_for_sites(
    sites_df: pd.DataFrame,
    times: Sequence,
    outdir: str = PATH_STORAGE / "era5_6h_out",
    download: bool = False,
    variables: Sequence[str] = DEFAULT_STD_VARS,
) -> pd.DataFrame:
    """
    在给定站点（site, lon, lat）和给定 time 序列上，提取 ERA5 单层 6h 气象变量。

    参数
    ----
    sites_df : pd.DataFrame
        至少包含列：
            - site : 站点 ID（字符串或数字，若缺失会自动生成 0..N-1）
            - lon  : 经度（单位度，[-180,180] 或 [0,360]）
            - lat  : 纬度（单位度，[-90,90]）
        可选：
            - alt  : 高程（米），若无则统一设为 0.0，用于输出（当前函数最终不返回 alt）。

    times : Sequence
        任意 datetime-like 序列（list[datetime], list[str], DatetimeIndex 等），
        约定为 **UTC**。若为带时区的 DatetimeIndex，会自动转换到 UTC 并去掉时区。

    outdir : str
        输出目录，用于存放/查找 ERA5 原始 .nc 文件：
            outdir/raw/single_levels/

    download : bool
        若为 True，则对缺失的月份尝试通过 cdsapi 下载（每月一个文件）。

        注意：当前版本仍然会读取 outdir/raw/single_levels/ 下 **所有** 真正的 .nc，
        然后在 time 维上做筛选；因此建议目录里只放你关心年份的 ERA5 文件。

    variables : Sequence[str]
        想要提取的“标准变量名”列表，比如 ("msl", "t2m", "tcwv")。
        标准名必须出现在 VAR_NAME_MAP 的 key 中。

    返回
    ----
    df : pd.DataFrame
        长表形式的数据框，至少包含：
            - time : 时间（UTC）
            - site : 站点 ID
            - 你在 variables 中指定的变量列（如果该变量在数据中存在）

        不再包含 lon, lat, alt 等站点信息列，仅保留最小必要变量。

    依赖
    ----
    - numpy
    - pandas
    - xarray
    - netCDF4 或 h5netcdf
    - cdsapi（仅在 download=True 时需要）
    """

    # ---- 处理变量列表：标准名 -> ERA5 名 ----
    variables = tuple(variables)
    era5_vars_to_download = [
        VAR_NAME_MAP[v] for v in variables if v in VAR_NAME_MAP
    ]
    if not era5_vars_to_download:
        raise ValueError(f"variables={variables} 全部不在 VAR_NAME_MAP 中。")

    if not try_import("xarray"):
        raise SystemExit(
            "需要安装 xarray/netcdf4/h5netcdf：\n"
            "    pip install xarray netCDF4 h5netcdf dask"
        )

    import xarray as xr

    # ---- 处理时间序列 ----
    if times is None or len(times) == 0:
        raise ValueError("times 不能为空。")

    times = pd.to_datetime(times)
    if isinstance(times, pd.DatetimeIndex):
        if times.tz is not None:
            times = times.tz_convert("UTC").tz_localize(None)
    else:
        # 普通 list -> DatetimeIndex
        times = pd.to_datetime(times)
        if times.tz is not None:
            times = times.tz_convert("UTC").tz_localize(None)

    times = pd.DatetimeIndex(times).sort_values()
    t_min, t_max = times.min(), times.max()
    print(f"[INFO] 目标时间范围: {t_min} ~ {t_max} (UTC)")

    # ---- 准备站点表 ----
    base = sites_df.copy()
    if "lon" not in base.columns or "lat" not in base.columns:
        raise SystemExit("sites_df 必须至少包含 lon, lat 列。")
    if "alt" not in base.columns:
        base["alt"] = 0.0
    if "site" not in base.columns:
        base["site"] = np.arange(len(base))

    # ---- 准备文件目录 ----
    outdir = ensure_dir(outdir)
    raw_dir = ensure_dir(os.path.join(outdir, "raw"))
    raw_sl = ensure_dir(os.path.join(raw_dir, "single_levels"))

    # ---- （可选）下载缺失月份 ----
    if download:
        if not try_import("cdsapi"):
            raise SystemExit("需要安装 cdsapi：pip install cdsapi")

        # 根据 times 推断需要的 year-month
        year_months = sorted({(t.year, t.month) for t in times})
        print("[INFO] 需要的 ERA5 月份：", year_months)

        for year, month in year_months:
            out_nc = os.path.join(raw_sl, f"sl_{year}_{month:02d}.nc")
            if not os.path.exists(out_nc):
                cds_retrieve_single_level(
                    year,
                    month,
                    out_nc,
                    variables=era5_vars_to_download,
                    times_6h=TIMES_6H,
                )
            else:
                print(f"[SKIP] 已存在，复用 {os.path.basename(out_nc)}")
    else:
        print("[INFO] 未开启 download，将仅复用本地 raw/single_levels/ 下已有 .nc")

    # ---- 兜底：处理“伪 .nc”压缩包 ----
    ensure_unpacked_in_folder(raw_sl)

    # ---- 收集所有真正的 NetCDF 文件 ----
    sl_files = []
    for fn in sorted(os.listdir(raw_sl)):
        if not fn.lower().endswith(".nc"):
            continue
        path = os.path.join(raw_sl, fn)
        if looks_like_netcdf(path):
            sl_files.append(path)

    if len(sl_files) == 0:
        raise SystemExit(f"{raw_sl} 下没有可读的 NetCDF (.nc) 文件。")

    print(f"[INFO] 将读取的单层 ERA5 文件数: {len(sl_files)}")

    # ---- 打开 ERA5 数据 ----
    engines = ["netcdf4", "h5netcdf", None]
    ds_sl = None
    last_err = None
    for eng in engines:
        try:
            ds_sl = xr.open_mfdataset(
                sl_files,
                combine="by_coords",
                engine=eng,
                combine_attrs="override",
                preprocess=harmonize_dims,  # 在这里统一 time/expver
                chunks={"time": 124},       # 31天×4步，和原脚本一致
            )
            print(f"[INFO] 使用 xarray.engine = {eng} 打开 ERA5 成功。")
            break
        except Exception as e:
            last_err = e
            print(f"[WARN] 使用 engine={eng} 打开失败：{e}")

    if ds_sl is None:
        raise SystemExit(f"Failed to open single-level datasets: {last_err}")

    # 按时间排序，方便后续最近时间匹配
    if "time" in ds_sl.dims:
        ds_sl = ds_sl.sortby("time")

    # ---- 提取核心变量 & 只保留关心的时间 ----
    core = build_core_dataset(ds_sl)
    # 最近时间匹配（nearest）
    core_sel = core.sel(time=times, method="nearest")

    # ---- 空间插值到站点 ----
    sampled_ds, lat_name, lon_name = sample_timeseries_at_sites(core_sel, base)

    # ---- 转为 DataFrame ----
    df = sampled_ds.to_dataframe().reset_index()

    # sampled_ds.to_dataframe() 会把 lat/lon 作为列名（如 latitude/longitude），删除之
    drop_cols = [c for c in (lat_name, lon_name) if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # 只保留需要的站点列（避免重复）。这里仅用 site，不合并 lon/lat/alt。
    site_cols = ["site"]
    site_cols = [c for c in site_cols if c in base.columns]
    site_info = base[site_cols].drop_duplicates(subset=["site"])

    # 合并站点信息（这里只会带入 site，本身已在 df 中，因此不会新增列）
    df = df.merge(site_info, on="site", how="left")

    # 排序 & 重置索引
    df = df.sort_values(["site", "time"]).reset_index(drop=True)

    # 只保留 time, site 和指定的气象变量
    keep_met_vars = [v for v in variables if v in df.columns]
    missing_vars = [v for v in variables if v not in df.columns]
    if missing_vars:
        print(f"[WARN] 下列变量在数据中不存在，将不出现在结果中: {missing_vars}")

    cols = ["time", "site"] + keep_met_vars

    return df[cols]


@flow.node()
def metvar(location, times, variables=DEFAULT_STD_VARS):
    """
    flow 节点封装：提取给定站点和时间序列的 ERA5 单层 6h 气象变量。

    参数
    ----
    location : pd.DataFrame
        站点表，同 extract_era5_6h_timeseries_for_sites 的 sites_df。
    times : Sequence
        时间序列。
    variables : Sequence[str]
        标准变量名，比如 ("msl", "t2m", "tcwv")。
    """
    return extract_era5_6h_timeseries_for_sites(
        location,
        times,
        outdir=PATH_STORAGE / "era5_6h_out",
        download=False,  # 本地已有 .nc 就设 False；否则 True 会用 cdsapi 下载缺失月份
        variables=variables,
    )


# ---------------- 可选：作为脚本运行时的小提示 ----------------
if __name__ == "__main__":
    print(
        "本文件定义了函数 `extract_era5_6h_timeseries_for_sites`，\n"
        "建议在 Python 中通过 import 调用。\n\n"
        "示例：\n"
        "  from era5_timeseries import extract_era5_6h_timeseries_for_sites\n"
        "  df = extract_era5_6h_timeseries_for_sites(\n"
        "           sites_df, times,\n"
        "           download=False,\n"
        "           variables=('msl', 't2m', 'tcwv'),\n"
        "       )\n"
    )
