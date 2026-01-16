from typing import List, Tuple

# !/usr/bin/env python3
import gzip
from datetime import datetime, timedelta

import numpy as np
from sklearn.neighbors import BallTree

from trop.reference.location import ngl_station

import time
from pathlib import Path

import requests
from requests.exceptions import SSLError, ProxyError, RequestException

from trop import PATH_STORAGE
from node import logger
from node import gather

from trop import flow
import pandas as pd

import zipfile
from node import track

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "NGL-Fetcher/1.0"})


def _head_exists(url: str) -> bool:
    try:
        return SESSION.head(url, timeout=15, allow_redirects=True).status_code != 404
    except RequestException as e:
        logger.warning(f"HEAD check failed for {url}: {e}")
        return False


@flow.node(workers=8)
def raw_ngl(year: int, site: str):
    out_dir = PATH_STORAGE / "raw_ngl"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{site}.{year}.trop.zip"

    def _is_valid_zip(path: Path) -> bool:
        try:
            with zipfile.ZipFile(path, 'r') as zf:
                bad = zf.testzip()
                if bad:
                    logger.error(f"Corrupt member in {path}: {bad}")
                    return False
            return True
        except zipfile.BadZipFile as e:
            logger.error(f"BadZipFile for {path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error validating {path}: {e}")
            return False

    if out_path.exists():
        if _is_valid_zip(out_path):
            return out_path
        logger.warning(f"Removing corrupted file {out_path}")
        out_path.unlink()

    url = f"https://geodesy.unr.edu/gps_timeseries/trop/{site}/{site}.{year}.trop.zip"
    if not _head_exists(url):
        logger.info(f"URL not found: {url}")
        return "not_found"

    backoff, max_tries = 2, 6
    for attempt in range(1, max_tries + 1):
        try:
            logger.info(f"[{attempt}/{max_tries}] Downloading {url}")
            resp = SESSION.get(url, stream=True, timeout=60)
            resp.raise_for_status()
            with out_path.open("wb") as fp:
                for chunk in resp.iter_content(1 << 15):
                    fp.write(chunk)

            if _is_valid_zip(out_path):
                logger.info(f"Downloaded and validated: {out_path}")
                return out_path

            logger.warning(f"Downloaded file corrupted, retrying…")
            out_path.unlink()

        except SSLError as e:
            logger.warning(f"SSL error ({e}), retrying in {backoff}s")
        except ProxyError as e:
            logger.warning(f"Proxy error ({e}), retrying in {backoff}s")
        except RequestException as e:
            logger.warning(f"Download error ({e}), retrying in {backoff}s")

        time.sleep(backoff)
        backoff *= 2

    logger.error(f"Failed to fetch a valid file after {max_tries} attempts: {url}")
    return "fail_to_fetch"


@flow.node()
def ztd_ngl(raw_ngl) -> pd.DataFrame:
    if not isinstance(raw_ngl, Path):
        return pd.DataFrame()
    times, ztds, sigmas = [], [], []
    with zipfile.ZipFile(raw_ngl) as zf:
        for inner in track(zf.namelist()):
            with zf.open(inner) as raw, gzip.open(raw, "rt", encoding="utf8") as trop:
                in_solution = False
                for line in trop:
                    if line.startswith("+TROP/SOLUTION"):
                        in_solution = True
                        continue
                    if line.startswith("-TROP/SOLUTION"):
                        in_solution = False
                        continue
                    if not in_solution:
                        continue
                    cols = line.split()
                    if (
                            len(cols) < 4
                            or cols[0].startswith("*")
                            or ":" not in cols[1]
                    ):
                        continue

                    yy, doy, sec = cols[1].split(":")
                    sec = int(sec)
                    if sec % 3600:
                        continue

                    yr = 2000 + int(yy)

                    try:
                        times.append(
                            datetime(yr, 1, 1)
                            + timedelta(days=int(doy) - 1, seconds=sec)
                        )
                        ztds.append(float(cols[2]))
                        sigmas.append(float(cols[3]))
                    except ValueError:
                        logger.warning(f"Bad line {inner}: {line}")
                        continue

    df = pd.DataFrame(
        {
            "time": times,
            "ztd_gnss": ztds,
            "ztd_gnss_sigma": sigmas,
        }
    )
    return df


@flow.node()
def ztd_ngl_region(year:int, ngl_station) -> pd.DataFrame:
    logger.info(f"rows extracted")
    sites = list(ngl_station.site)
    logger.info(sites)
    task = gather([ztd_ngl(raw_ngl(year, site)) for site in sites])
    task.get()
    df_list = []
    for site in track(sites):
        logger.info(f"start rows extracted")
        df = ztd_ngl(raw_ngl(year, site)).get()
        logger.info(f"{site}:{len(df)}")
        df['site'] = site
        df_list.append(df)
    df = pd.concat(df_list)
    logger.info(f"{len(df)} rows extracted")
    return df


# Earth radius in km
R = 6371.0


def select_sparse_stations(df, min_distance=0):
    # Convert lat/lon to radians
    coords = np.deg2rad(df[['lat', 'lon']].values)

    # Build a BallTree using Haversine metric
    tree = BallTree(coords, metric='haversine')

    # Convert km to radians
    min_dist_rad = min_distance / R

    selected = []
    excluded = np.zeros(len(df), dtype=bool)

    for i in range(len(df)):
        if not excluded[i]:
            selected.append(i)
            # Find all points within 50 km of this one
            ind = tree.query_radius(coords[i:i + 1], r=min_dist_rad)[0]
            excluded[ind] = True  # Exclude nearby points

    return df.iloc[selected]['site'].tolist()

@flow.node()
def ztd_ngl_dataset(ztd_ngl_region: pd.DataFrame,
                    ngl_station: pd.DataFrame,
                    min_distance: float,
                    sigma_quantile: float,
                    site_threshold: float,
                    time_threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame,dict, list, list, list]:
    df = ztd_ngl_region

    sigma_threshold = df['ztd_gnss_sigma'].quantile(sigma_quantile)
    df=df[df['ztd_gnss_sigma'] <= sigma_threshold].copy()

    nt = df['time'].nunique()
    ns = df['site'].nunique()

    good_sites = df.groupby('site')['time'].nunique().gt(site_threshold * nt)
    good_times = df.groupby('time')['site'].nunique().gt(time_threshold * ns)

    good_sites_list=good_sites[good_sites].index.tolist()
    good_times_list=good_times[good_times].index.tolist()

    filter_info = {
        'sigma_threshold':sigma_threshold,
        'site': {
            'total_times': int(nt),
            'min_times': site_threshold * nt,
            'remain_rate': good_sites.sum() / len(good_sites),
        },
        'time': {
            'total_sites': int(ns),
            'min_sites': time_threshold * ns,
            'remain_rate': good_times.sum() / len(good_times),
        },
    }

    df_filtered = df[
        df['site'].isin(good_sites_list) &
        df['time'].isin(good_times_list)
        ]

    location = ngl_station[ngl_station['site'].isin(good_sites_list)]

    if min_distance>0:
        sparse_site_lists = select_sparse_stations(location,min_distance=min_distance)
        print(sparse_site_lists)
        df_filtered = df_filtered[df_filtered['site'].isin(sparse_site_lists)]
        location=location[location['site'].isin(sparse_site_lists)]
    else:
        sparse_site_lists=good_sites_list

    return df_filtered, filter_info ,location,good_sites_list,sparse_site_lists, good_times_list





if __name__ == "__main__":
    print(ztd_ngl_dataset())
    print(ztd_ngl_dataset().get())
