"""
ParallelDownloader — high‑throughput, resilient file downloader
==============================================================

Features
--------
* **Multi‑thread segmented downloading** via HTTP *Range* requests.
* Automatic *Accept‑Ranges* capability detection and graceful fallback.
* Persistent connection pools using ``requests.Session`` + ``HTTPAdapter``.
* Live progress bar powered by **rich** with transfer speed & ETA.
* Configurable chunk size, concurrency, retry limits and back‑off.
* Graceful Ctrl‑C cancellation and clean resource shutdown.

Usage
-----
```python
from parallel_downloader import ParallelDownloader

url = "https://example.com/very_large_file.iso"
downloader = ParallelDownloader(url, threads=8, target_path="file.iso")
path = downloader.download()  # blocking call
print(f"Saved to {path}")
```
Install requirements first:
```
pip install requests rich
```
"""
from __future__ import annotations

import math
import os
import signal
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

# ------------------------------
# helpers
# ------------------------------

_DEFAULT_CHUNK = 1 << 16  # 64 KiB per read


def _get_head_info(url: str, session: requests.Session, timeout: Optional[float]) -> tuple[int, bool]:
    """Return (content_length, accept_ranges)"""
    head = session.head(url, allow_redirects=True, timeout=timeout)
    head.raise_for_status()
    size = int(head.headers.get("Content-Length", "0"))
    accept = head.headers.get("Accept-Ranges", "none").lower() == "bytes"
    return size, accept


class _SegmentThread(threading.Thread):
    """Worker responsible for downloading a single byte‑range to file."""

    def __init__(
        self,
        url: str,
        start: int,
        end: int,
        session: requests.Session,
        file_path: Path,
        task_id: TaskID,
        progress: Progress,
        stop_event: threading.Event,
        chunk_size: int,
        timeout: Optional[float],
        retries: int,
        backoff: float,
    ) -> None:
        super().__init__(daemon=True)
        self.url = url
        self.start_byte = start
        self.end_byte = end
        self.session = session
        self.file_path = file_path
        self.task_id = task_id
        self.progress = progress
        self.stop_event = stop_event
        self.chunk_size = chunk_size
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff

    def run(self) -> None:  # noqa: WPS231
        headers = {"Range": f"bytes={self.start_byte}-{self.end_byte}"}
        # simple retry loop
        attempt = 0
        while attempt <= self.retries and not self.stop_event.is_set():
            try:
                with self.session.get(
                    self.url,
                    headers=headers,
                    stream=True,
                    timeout=self.timeout,
                ) as resp:
                    resp.raise_for_status()
                    with open(self.file_path, "r+b") as fp:
                        fp.seek(self.start_byte)
                        for chunk in resp.iter_content(chunk_size=self.chunk_size):
                            if self.stop_event.is_set():
                                return
                            if chunk:
                                fp.write(chunk)
                                self.progress.update(self.task_id, advance=len(chunk))
                    return  # success
            except (requests.RequestException, OSError):
                attempt += 1
                if attempt > self.retries:
                    self.stop_event.set()
                    return
                time.sleep(self.backoff * attempt)


class ParallelDownloader:
    """High‑level API for fast, segmented downloads with progress display."""

    def __init__(
        self,
        url: str,
        *,
        target_path: str | os.PathLike | None = None,
        threads: int = 8,
        chunk_size: int = _DEFAULT_CHUNK,
        timeout: float | None = None,
        retries: int = 3,
        backoff: float = 1.0,
    ) -> None:
        self.url = url
        self.threads = max(1, threads)
        self.chunk_size = chunk_size
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.stop_event = threading.Event()

        # choose filename
        default_name = Path(url).name or "download.bin"
        self.file_path = Path(target_path) if target_path else Path.cwd() / default_name

        # global session
        self.session = requests.Session()
        adapter = HTTPAdapter(pool_connections=self.threads, pool_maxsize=self.threads)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    # --------------------------
    # public API
    # --------------------------
    def download(self) -> Path:
        """Download the file and return the output path. Blocking."""
        size, supports_range = _get_head_info(self.url, self.session, self.timeout)
        if size <= 0:
            raise RuntimeError("Unable to determine Content‑Length; aborting.")

        if not supports_range or self.threads == 1:
            # fall back to single‑stream download with progress
            return self._single_stream(size)
        return self._multi_stream(size)

    # --------------------------
    # internal helpers
    # --------------------------
    def _single_stream(self, size: int) -> Path:
        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("Downloading", total=size)
            with self.session.get(self.url, stream=True, timeout=self.timeout) as resp, open(
                self.file_path,
                "wb",
            ) as fp:
                resp.raise_for_status()
                for chunk in resp.iter_content(chunk_size=self.chunk_size):
                    if self.stop_event.is_set():
                        break
                    if chunk:
                        fp.write(chunk)
                        progress.update(task, advance=len(chunk))
        return self.file_path

    def _multi_stream(self, size: int) -> Path:
        # pre‑allocate file
        with open(self.file_path, "wb") as fp:
            fp.truncate(size)
        segment = math.ceil(size / self.threads)

        # Ctrl‑C handler
        def _sigint_handler(signum, frame):  # noqa: WPS430
            self.stop_event.set()
        with suppress(ValueError):
            signal.signal(signal.SIGINT, _sigint_handler)

        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("Downloading", total=size)
            # spawn segment workers
            threads: list[_SegmentThread] = []
            for i in range(self.threads):
                start = i * segment
                end = min(start + segment - 1, size - 1)
                t = _SegmentThread(
                    self.url,
                    start,
                    end,
                    self.session,
                    self.file_path,
                    task,
                    progress,
                    self.stop_event,
                    self.chunk_size,
                    self.timeout,
                    self.retries,
                    self.backoff,
                )
                threads.append(t)
                t.start()
            for t in threads:
                t.join()

        if self.stop_event.is_set():
            raise RuntimeError("Download interrupted or failed.")
        return self.file_path

    # --------------------------
    # context manager helpers
    # --------------------------
    def __enter__(self):  # noqa: D401
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: D401
        self.stop_event.set()
        self.session.close()
        return False  # do not suppress exceptions

if __name__ == "__main__":
    url = (
        "https://apps.ecmwf.int/api/streaming/private/"
        "blue/02/20250622-0430/50/"
        "_grib2netcdf-bol-webmars-private-svc-blue-"
        "001-4a73a881a8d5eead47db9eff2f9935a4-_0bEc4.nc"
    )
    downloader = ParallelDownloader(url, threads=64, target_path="bigfile.nc")
    downloader.download()

