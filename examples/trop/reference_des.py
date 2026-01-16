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
def _compute_regions(countries: gpd.GeoDataFrame) -> dict[str, gpd.GeoDataFrame]:
    """
    Prepare 'us' and 'eu' region GeoDataFrames keyed by name.
    """
    regs: dict[str, gpd.GeoDataFrame] = {}

    # US contiguous (largest polygon)
    us = countries[countries["ISO_A2"] == "US"]
    regs["us"] = _top_n_blocks(us, n=1).assign(region="us")

    # Europe minus Russia, take top 2
    eu = countries[countries["CONTINENT"] == "Europe"]
    ru = countries[countries["ISO_A2"] == "RU"]
    eu_minus_ru = gpd.overlay(eu, ru, how="difference")
    regs["eu"] = _top_n_blocks(eu_minus_ru, n=2).assign(region="eu")

    # Australia (entire country as one polygon)
    au = countries[countries["ISO_A2"] == "AU"]
    regs["au"] = _top_n_blocks(au, n=1).assign(region="au")


    return regs


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
        union_poly = rg.geometry.union_all()
        mask = pts.within(union_poly)
        df.loc[mask, "region"] = name

    return df


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
@flow.node()
def ngl_station(region='all') -> pd.DataFrame:
    """
    Return station DataFrame with 'region' column, using cache if available.
    """

    # load raw, boundaries, compute + assign
    raw_df = _load_raw_stations()
    countries = _load_country_boundaries()
    regions = _compute_regions(countries)
    proc_df = _assign_regions(raw_df, regions)
    proc_df.to_csv(PATH_NGL_STATION_INFO, index=False)

    if region == 'all':
        return proc_df
    elif region == 'us':
        return proc_df[proc_df.region == 'us']
    elif region == 'eu':
        return proc_df[proc_df.region == 'eu']
    elif region == 'head':
        return proc_df.head()
    else:
        raise ValueError(f"Unknown region {region}")


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
    df['site'] = [f"site_{i:07d}" for i in range(len(df))]

    # 返回指定列顺序的 DataFrame
    return df[['site', 'lon', 'lat']]


@flow.node()
def grid(spacing=0.25,region='all') -> pd.DataFrame:
    """
    Return station DataFrame with 'region' column, using cache if available.
    """

    # load raw, boundaries, compute + assign
    raw_df = _load_grid(spacing=spacing)
    countries = _load_country_boundaries()
    regions = _compute_regions(countries)
    proc_df = _assign_regions(raw_df, regions)
    proc_df.to_csv(PATH_NGL_STATION_INFO, index=False)

    if region == 'all':
        return proc_df
    elif region == 'us':
        return proc_df[proc_df.region == 'us']
    elif region == 'eu':
        return proc_df[proc_df.region == 'eu']
    elif region == 'head':
        return proc_df.head()
    else:
        logger.error(f"Unknown region {region}")
        return None


ngl_us=ngl_station(region='us')
ngl=ngl_station()
grid_us=grid(spacing=0.25,region='us')

if __name__ == "__main__":
    ngl_us.get()
    ngl.get()
    grid_us.get()