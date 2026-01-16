#!/usr/bin/env python3
"""
ngl_station_info.py

Provides `ngl_station_info` DataFrame: station metadata + region labels.
On first import, downloads raw station info and boundary shapefiles, computes
regions, assigns labels and caches to disk. Next imports read cached file only.
"""

from loguru import logger
from pathlib import Path
import zipfile

import pandas as pd
import geopandas as gpd
import requests

from trop import PROJ_ROOT,flow

# ---------------------------------------------------------------------
# Paths and URLs
# ---------------------------------------------------------------------
SITE_DIR = PROJ_ROOT / "reference" / "location" / "site"
SITE_DIR.mkdir(parents=True, exist_ok=True)
PATH_NGL_STATION_INFO_RAW = SITE_DIR / "ngl_station_info_raw.csv"
PATH_NGL_STATION_INFO = SITE_DIR / "ngl_station_info.csv"

BOUNDARY_DIR = PROJ_ROOT / "reference" / "location" / "boundary"
BOUNDARY_DIR.mkdir(parents=True, exist_ok=True)

LAND_URL = "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_land.zip"
COUNTRIES_URL = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"

LAND_ZIP = BOUNDARY_DIR / "ne_10m_land.zip"
LAND_DIR = BOUNDARY_DIR / "ne_10m_land"
COUNTRIES_ZIP = BOUNDARY_DIR / "ne_10m_admin_0_countries.zip"
COUNTRIES_DIR = BOUNDARY_DIR / "ne_10m_admin_0_countries"


# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------
def _download_and_unzip(url: str, zip_path: Path, extract_to: Path) -> None:
    """
    Download a ZIP if missing and extract it.
    """
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        logger.info(f"Downloading: {url}")
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                if chunk:
                    f.write(chunk)
    extract_to.mkdir(parents=True, exist_ok=True)
    if not any(extract_to.iterdir()):
        logger.info(f"Extracting: {zip_path} -> {extract_to}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)


def _top_n_blocks(
        gdf: gpd.GeoDataFrame,
        n: int = 1,
        proj_epsg: int = 3395,
) -> gpd.GeoDataFrame:
    """
    Dissolve -> Explode -> Project -> Take top n by area -> back CRS
    """
    pieces = (
        gdf.dissolve()
        .explode(index_parts=False)
        .reset_index(drop=True)
    )
    proj = pieces.to_crs(epsg=proj_epsg)
    top_idx = proj.area.nlargest(n).index
    return pieces.loc[top_idx].to_crs(gdf.crs).reset_index(drop=True)
def _top_n_blocks_ea(
    gdf: gpd.GeoDataFrame, n: int = 1, proj_epsg: int = 6933
) -> gpd.GeoDataFrame:
    pieces = (
        gdf.dissolve()
        .explode(index_parts=False)
        .reset_index(drop=True)
    )
    if pieces.empty:
        return pieces
    proj = pieces.to_crs(epsg=proj_epsg)
    top_idx = proj.area.nlargest(n).index
    return pieces.loc[top_idx].to_crs(gdf.crs).reset_index(drop=True)


def _filter_blocks_by_proximity(
    gdf: gpd.GeoDataFrame, max_km: float, proj_epsg: int = 3035
) -> gpd.GeoDataFrame:
    """
    以投影到米制(默认欧洲等面积 3035)后的最大块为“主陆块”，
    只保留与其距离 <= max_km 的块。
    """
    pieces = (
        gdf.dissolve()
        .explode(index_parts=False)
        .reset_index(drop=True)
    )
    if pieces.empty:
        return pieces

    proj = pieces.to_crs(epsg=proj_epsg)
    areas = proj.area
    anchor_idx = areas.idxmax()          # 最大块 = 主陆块
    anchor_geom = proj.iloc[anchor_idx].geometry.buffer(max_km * 1000.0)
    keep_mask = proj.intersects(anchor_geom)
    return pieces.loc[keep_mask.values].to_crs(gdf.crs).reset_index(drop=True)

def _filter_blocks_by_area(
    gdf: gpd.GeoDataFrame,
    min_km2: float = 100.0,
    proj_epsg: int = 6933,
) -> gpd.GeoDataFrame:
    """
    Dissolve -> Explode -> Project(equal-area) -> keep blocks with area >= min_km2 -> back CRS
    """
    pieces = (
        gdf.dissolve()
        .explode(index_parts=False)
        .reset_index(drop=True)
    )
    if pieces.empty:
        return pieces

    proj = pieces.to_crs(epsg=proj_epsg)     # 等面积投影，面积单位 m²
    areas_km2 = proj.area / 1e6
    kept_idx = areas_km2[areas_km2 >= float(min_km2)].index

    if len(kept_idx) == 0:                   # 兜底：至少保留最大块
        kept_idx = [proj.area.idxmax()]

    return pieces.loc[kept_idx].to_crs(gdf.crs).reset_index(drop=True)




# ---------------------------------------------------------------------
# Raw Station Data
# ---------------------------------------------------------------------
def _load_raw_stations() -> pd.DataFrame:
    """
    Download and cache raw NGL station list.
    """
    if PATH_NGL_STATION_INFO_RAW.exists():
        logger.info(f"Loading raw stations from: {PATH_NGL_STATION_INFO_RAW}")
        return pd.read_csv(PATH_NGL_STATION_INFO_RAW)

    logger.info("Downloading raw station info...")
    url = "https://geodesy.unr.edu/NGLStationPages/DataHoldings.txt"
    df = pd.read_csv(url, sep=r"\s+", on_bad_lines="skip")
    df = (
        df.rename(columns={
            "Sta": "site",
            "Lat(deg)": "lat",
            "Long(deg)": "lon",
            "Hgt(m)": "alt",
            "X(m)": "x",
            "Y(m)": "y",
            "Z(m)": "z",
        })
        .loc[:, ["site", "lat", "lon", "alt", "x", "y", "z"]]
    )
    df.to_csv(PATH_NGL_STATION_INFO_RAW, index=False)
    return df


# ---------------------------------------------------------------------
# Boundary Data
# ---------------------------------------------------------------------
def _load_country_boundaries() -> gpd.GeoDataFrame:
    """
    Download + unzip Natural Earth country boundaries, return GeoDataFrame.
    """
    _download_and_unzip(LAND_URL, LAND_ZIP, LAND_DIR)
    _download_and_unzip(COUNTRIES_URL, COUNTRIES_ZIP, COUNTRIES_DIR)
    shp = next(COUNTRIES_DIR.glob("*.shp"))
    return gpd.read_file(shp).to_crs("EPSG:4326")


# ---------------------------------------------------------------------
# Compute Regions
# ---------------------------------------------------------------------
REGION_DEFS = {
    # 0.03° ≈ 3 km（纬向），足够抹平边界毛刺，基本不会跨海连陆
    # max_km_from_main 控制是否保留“离本土很远”的块
    "us": {"iso": "US",
           "exclude_area_less_than": 0,
           "GAP_DEG": 0.25,
           "max_km_from_main": 800},    # 保留阿拉斯加？否（>2000 km）；夏威夷？否

    "eu": {"continent": "Europe",
           "exclude_iso": "RU",
           "exclude_area_less_than": 0,
           "GAP_DEG": 0.25,
           "max_km_from_main": 1000},   # 保留冰岛（~1000–1200 km），排除加那利/北非

    "au": {"iso": "AU",
           "exclude_area_less_than": 0,
           "GAP_DEG": 0.25,
           "max_km_from_main": 1000},   # 保留塔斯马尼亚、菲利普等，排除诺福克/科科斯等遥远岛

    "jp": {"iso": "JP",
           "exclude_area_less_than": 0,
           "GAP_DEG": 0.25,
           "max_km_from_main": 1000},   # 保留琉球/与那国（~1000 km），基本排除小笠原（>1000 km，按需调整）
}





def _resolve_regions(
    countries: gpd.GeoDataFrame,
    return_area: bool = False,
    proj_epsg: int = 6933,   # equal-area by default
    units: str = "km2",
):
    """
    构造各区域多边形集合：
      1) 先按照 continent/iso/exclude_iso 过滤国家；
      2) 用等面积投影按面积阈值剔除小碎块（exclude_area_less_than, km²）；
      3) 可选：按等面积面积取前 N 个块（top_n_blocks）；
      4) 可选：只保留距离主陆块 ≤ max_km_from_main 的块（km）。
    """
    regs: dict[str, gpd.GeoDataFrame] = {}

    for name, cfg in REGION_DEFS.items():
        # 1) 选取区域国家
        df = countries
        if "continent" in cfg:
            df = df[df.CONTINENT == cfg["continent"]]
        if "iso" in cfg:
            df = df[df.ISO_A2 == cfg["iso"]]
        if "exclude_iso" in cfg:
            ex = countries[countries.ISO_A2 == cfg["exclude_iso"]]
            if not ex.empty and not df.empty:
                df = gpd.overlay(df, ex, how="difference")

        # 2) 面积阈值（km²），先剔除很小的碎块
        min_km2 = float(cfg.get("exclude_area_less_than", 0))
        if min_km2 > 0:
            blocks = _filter_blocks_by_area(df, min_km2=min_km2, proj_epsg=proj_epsg)
        else:
            blocks = (
                df.dissolve()
                  .explode(index_parts=False)
                  .reset_index(drop=True)
            )

        # 3) （可选）等面积 top-n
        tn = cfg.get("top_n_blocks", None)
        if isinstance(tn, int) and tn > 0 and not blocks.empty:
            _proj = blocks.to_crs(epsg=proj_epsg)
            top_idx = _proj.area.nlargest(tn).index
            blocks = blocks.loc[top_idx].reset_index(drop=True)

        # 4) （可选）主陆块邻近过滤（km）
        max_km = float(cfg.get("max_km_from_main", 0))
        if max_km > 0 and not blocks.empty:
            # 投到米制投影做距离缓冲；3035 在欧洲最好，但对全球也可用
            _proj = blocks.to_crs(epsg=3035)
            anchor_idx = _proj.area.idxmax()                       # 最大块作为主陆块
            anchor_buf = _proj.iloc[anchor_idx].geometry.buffer(max_km * 1000.0)
            keep_mask = _proj.intersects(anchor_buf)
            blocks = blocks.loc[keep_mask.values].reset_index(drop=True)

        regs[name] = blocks.assign(region=name)

    if not return_area:
        return regs

    # 需要返回面积统计
    if units not in {"m2", "km2"}:
        raise ValueError("units must be 'm2' or 'km2'")
    denom = 1e6 if units == "km2" else 1.0

    areas: dict[str, float] = {}
    for name, gdf_region in regs.items():
        if gdf_region.empty:
            areas[name] = 0.0
        else:
            a_m2 = float(gdf_region.to_crs(epsg=proj_epsg).area.sum())
            areas[name] = a_m2 / denom

    return regs, areas







# ---------------------------------------------------------------------
# Assign Regions
# ---------------------------------------------------------------------
def _assign_regions(
        stations: pd.DataFrame,
        regions: dict[str, gpd.GeoDataFrame],
        lon_col: str = "lon",
        lat_col: str = "lat",
) -> pd.DataFrame:
    """
    Label stations with region name if within any region polygon.
    """
    df = stations.copy()
    df[lon_col] = ((df[lon_col] + 180) % 360) - 180  # wrap lon
    pts = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326",
    )
    df["region"] = None

    for name, rg in regions.items():
        gap_deg = float(REGION_DEFS.get(name, {}).get("GAP_DEG", 0.0))
        # 兼容 union_all / unary_union
        try:
            union_poly = rg.geometry.union_all()
        except Exception:
            union_poly = rg.geometry.unary_union
        # 角度缓冲抹平细小裂缝，然后判定相交（包含边界点）
        union_poly = union_poly.buffer(gap_deg).buffer(0)
        mask = pts.intersects(union_poly)
        df.loc[mask, "region"] = name

    return df

def _build_regioned_df(loader_fn, cache_path, *loader_args, **loader_kwargs):
    if cache_path.exists():
        df = pd.read_csv(cache_path)
    else:
        df = loader_fn(*loader_args, **loader_kwargs)
        df.to_csv(cache_path, index=False)
    countries = _load_country_boundaries()
    regions = _resolve_regions(countries)
    return _assign_regions(df, regions)

def _filter_region(df, region):
    if region == "all":
        return df
    if region in REGION_DEFS:
        return df[df.region == region]
    if region == "head":
        return df.head()
    logger.error(f"Unknown region {region}")
    return pd.DataFrame()


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
@flow.node()
def ngl_station(region='all') -> pd.DataFrame:
    """
    Return station DataFrame with 'region' column, using cache if available.
    """

    df = _build_regioned_df(_load_raw_stations, PATH_NGL_STATION_INFO)
    return _filter_region(df, region)

@flow.node()
def ngl_station_extend(stations):
    import math
    import numpy as np
    import pandas as pd

    GRID = 0.25  # 格网间距

    def wrap_lon(lon):
        """经度规范到 [-180, 180)"""
        return ((lon + 180.0) % 360.0) - 180.0

    def four_corners(lat, lon, d=GRID):
        """返回包围 (lat, lon) 的 4 个格点 (LB, RB, LT, RT)"""
        lon = wrap_lon(lon)
        base_lat = math.floor(lat / d) * d
        base_lon = math.floor(lon / d) * d
        points = {
            'LB': (base_lat, base_lon),
            'RB': (base_lat, base_lon + d),
            'LT': (base_lat + d, base_lon),
            'RT': (base_lat + d, base_lon + d),
        }
        # 经度 wrap
        points = {k: (v[0], wrap_lon(v[1])) for k, v in points.items()}
        return points

    def make_gridpoints(stations):
        rows = []
        for _, r in stations.iterrows():
            corners = four_corners(float(r['lat']), float(r['lon']), GRID)
            for suffix, (la, lo) in corners.items():
                rows.append({
                    'site': f"{r['site']}{suffix}",
                    'lat': round(la, 6),
                    'lon': round(lo, 6),
                    'alt': r['alt'],
                    'x': np.nan,
                    'y': np.nan,
                    'z': np.nan,
                    'region': r['region']
                })
        return pd.DataFrame(rows, columns=stations.columns)

    # 假设原表是 stations
    # stations = pd.read_csv("stations.csv")

    gridpoints = make_gridpoints(stations)

    # 拼接
    combined = pd.concat([stations, gridpoints], ignore_index=True)

    # 保存或查看
    return combined


@flow.node()
def grid(spacing=0.25, region="all"):
    path = SITE_DIR / f"ngl_grid_{spacing}.csv"
    df = _build_regioned_df(_load_grid, path, spacing)
    return _filter_region(df, region)

import numpy as np
import pandas as pd

def _load_grid(spacing=0.25) -> pd.DataFrame:
    """
    生成全球经纬度等间隔格网点

    参数:
        spacing (float): 经纬度间隔，单位为度 (°)

    返回:
        pd.DataFrame: 包含三列 'site', 'lon', 'lat'，其中 lon, lat 单位为 °，site 为格网点名称
    """
    # 定义经度和纬度范围，包含两端点
    lons = np.arange(-180, 180 + spacing, spacing)
    lats = np.arange(-90, 90 + spacing, spacing)

    # 构建网格点列表
    coords = [(lon, lat) for lon in lons for lat in lats]

    # 转换为 DataFrame
    df = pd.DataFrame(coords, columns=['lon', 'lat'])

    # 生成 site 名称，可根据需求自定义格式
    len_df=len(df)
    df['site'] = [f"G{i:0{len(str(len_df))}d}" for i in range(len_df)]

    # 返回指定列顺序的 DataFrame
    return df[['site', 'lon', 'lat']]


ngl_us=ngl_station(region='us')
ngl_au=ngl_station(region='au')
ngl_jp=ngl_station(region='jp')
ngl=ngl_station()
grid_us=grid(spacing=0.25,region='us')

if __name__ == "__main__":
    ngl_us.get()
    ngl_au.get()
    ngl_jp.get()
    ngl.get()
    grid_us.get()