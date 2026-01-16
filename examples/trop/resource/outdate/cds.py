from datetime import timedelta, datetime
import os
import cdsapi
import pandas as pd
from pathlib import Path

class ERA5Downloader:
    def __init__(self, download_dir, interval='1h', request='ztd'):
        self.download_dir = download_dir
        self.interval = interval
        self.request = request
        self.urls_dict = {}

        home_dir = os.path.expanduser("~")
        cdsapirc_path = os.path.join(home_dir, ".cdsapirc")

        content = """
url: https://cds.climate.copernicus.eu/api
key: efe77d01-5f17-4278-880c-b44df1088ce0"""
        with open(cdsapirc_path, "w") as file:
            file.write(content)

    def retrieve_data(self, year, month, days, hours, path):
        if self.request == 'ztd':
            dataset = "reanalysis-era5-pressure-levels"
            request = {
                'product_type': ['reanalysis'],
                'variable': ['geopotential', 'specific_humidity', 'temperature'],
                'year': [str(year)],
                'month': [f'{month:02d}'],
                'day': [f'{day:02d}' for day in days],
                'time': [f'{hour:02d}:00' for hour in hours],
                'pressure_level': ['1', '2', '3', '5', '7', '10', '20', '30', '50', '70', '100', '125', '150', '175',
                                   '200', '225', '250', '300', '350', '400', '450', '500', '550', '600', '650', '700',
                                   '750', '775', '800', '825', '850', '875', '900', '925', '950', '975', '1000'],
                'data_format': 'netcdf',
                'download_format': 'unarchived',
            }
        else:
            raise NotImplementedError

        client = cdsapi.Client()
        client.retrieve(dataset, request, path)

    from joblib import Parallel, delayed

    def download_era5(self, start_date, end_date):
        if not isinstance(start_date, datetime):
            start_date = pd.to_datetime(start_date)
        if not isinstance(end_date, datetime):
            end_date = pd.to_datetime(end_date)

        current_date = start_date
        tasks = []

        while current_date <= end_date:
            year = current_date.year
            month = current_date.month

            if self.interval == '1h':
                days = [current_date.day]
                hours = [current_date.hour]
                next_date = current_date + timedelta(hours=1)
                filename = f'ERA5_{current_date.strftime("%Y%m%d_%H")}_{self.interval}.nc'
            elif self.interval == '1d':
                days = [current_date.day]
                hours = list(range(24))
                next_date = current_date + timedelta(days=1)
                filename = f'ERA5_{current_date.strftime("%Y%m%d")}_{self.interval}.nc'
            elif self.interval == '1m':
                days = list(range(1, 32))
                hours = list(range(24))
                next_month = current_date.replace(day=28) + timedelta(days=4)  # this will never fail
                next_date = next_month - timedelta(days=next_month.day - 1)
                filename = f'ERA5_{current_date.strftime("%Y%m")}_{self.interval}.nc'
            else:
                raise ValueError("Invalid interval. Use '1h', '1d', or '1m'.")

            tasks.append((year, month, days, hours, current_date, filename))
            current_date = next_date

        def process_task(year, month, days, hours, filename):
            try:
                path = Path(self.download_dir) / filename
                print(f"Retrieving: {filename} to {path}")
                self.retrieve_data(year, month, days, hours, path)
                print(f"{filename} has been donwloaded")
            except Exception as exc:
                print(f"Failed to download data for {year}-{month:02d}-{days[0]:02d} to {days[-1]:02d}: {exc}")

        from tqdm_joblib import ParallelPbar
        from joblib import delayed
        ParallelPbar('Downloading File')(n_jobs=32, backend='threading')(
            delayed(process_task)(year, month, days, hours, filename) for
            year, month, days, hours, current_date, filename in tasks)


if __name__ == '__main__':
    downloader = ERA5Downloader(interval='1h', download_dir=r'Z:\NWM\ERA5\global')
    start_date = datetime(2023, 1, 1, 0)
    end_date = datetime(2023, 1, 1, 1)
    downloader.download_era5(start_date, end_date)
