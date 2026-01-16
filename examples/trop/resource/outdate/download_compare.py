#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import threading
import subprocess
import asyncio

import requests
import aiohttp
import httpx

# —— 配置区 ——
URL = (
    "https://object-store.os-api.cci2.ecmwf.int:443/cci2-prod-cache-1/2025-07-10/c5860092d1f875a6e1038a83f7a64aa7.nc"

)
DURATION = 60  # 单次测试时长（秒）
THREADS = 16  # 并发线程数


# —— 方案 A：单流 requests ——
def requests_single():
    end_t = time.time() + DURATION
    total = 0
    # 不设置 timeout，避免 ReadTimeout
    s = requests.Session()
    # 池化连接
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=THREADS,
        pool_maxsize=THREADS
    )
    s.mount("https://", adapter)
    resp = s.get(URL, stream=True, timeout=None)
    for chunk in resp.iter_content(chunk_size=64 * 1024):
        if not chunk or time.time() >= end_t:
            break
        total += len(chunk)
    resp.close()
    return total


# —— 方案 B：并发请求（full-stream） ——
def requests_parallel():
    end_t = time.time() + DURATION
    counter = {"bytes": 0}
    lock = threading.Lock()

    def worker():
        sess = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=THREADS,
            pool_maxsize=THREADS
        )
        sess.mount("https://", adapter)
        while time.time() < end_t:
            r = sess.get(URL, stream=True, timeout=None)
            for chunk in r.iter_content(64 * 1024):
                if not chunk or time.time() >= end_t:
                    break
                with lock:
                    counter["bytes"] += len(chunk)
            r.close()

    threads = [threading.Thread(target=worker) for _ in range(THREADS)]
    for t in threads: t.start()
    for t in threads: t.join()
    return counter["bytes"]


# —— 方案 C：aiohttp 异步 ——
async def _aio_fetch(end_t, total):
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as sess:
        while time.time() < end_t:
            async with sess.get(URL) as r:
                async for chunk in r.content.iter_chunked(64 * 1024):
                    if time.time() >= end_t:
                        break
                    total[0] += len(chunk)


def aiohttp_async():
    end_t = time.time() + DURATION
    total = [0]
    asyncio.run(_aio_fetch(end_t, total))
    return total[0]


# —— 方案 D：httpx HTTP/2 ——
def httpx_http2():
    end_t = time.time() + DURATION
    total = 0
    client = httpx.Client(http2=True, timeout=None)
    while time.time() < end_t:
        with client.stream("GET", URL) as r:
            for chunk in r.iter_bytes(64 * 1024):
                if time.time() >= end_t:
                    break
                total += len(chunk)
    client.close()
    return total


# —— 方案 E：aria2c 子进程 ——
def aria2c_proc():
    fname = "aria2_tmp.nc"
    if os.path.exists(fname):
        os.remove(fname)
    cmd = [
        "aria2c", "-c",
        "--split=16", "--max-connection-per-server=2",
        "--file-allocation=none",
        "-d", ".", "-o", fname, URL
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    out, err = p.communicate()
    print(err.decode())
    time.sleep(DURATION)
    p.terminate()
    p.wait()
    return os.path.getsize(fname) if os.path.exists(fname) else 0

import sys
def aria2c_proc2():
    venv_scripts = os.path.dirname(sys.executable)
    aria2c_exe = os.path.join(venv_scripts, "aria2c.exe")
    print(aria2c_exe, os.path.isfile(aria2c_exe))

    import tempfile
    fname = "aria2_tmp.nc"
    cmd = [
        aria2c_exe,
        "-c",
        "--split=16",
        "--max-connection-per-server=16",
        "--file-allocation=none",
        "-d", os.path.dirname(fname) or ".",
        "-o", os.path.basename(fname),
        URL,
    ]
    subprocess.check_call(cmd)
    return os.path.getsize(fname)


# —— 主流程 ——
def main():
    funcs = [
        ("requests_single", requests_single),
        ("requests_parallel", requests_parallel),
        # ("aiohttp_async", aiohttp_async),
        # ("httpx_http2", httpx_http2),
        # ("aria2c_pro",aria2c_proc2),
        ("aria2c", aria2c_proc),
    ]
    print(f"每种方案测试 {DURATION} 秒 后的下载量与平均速度：\n")
    for name, fn in funcs:
        print(f"[{name}] 测试中 …")
        n = fn()
        mib = n / 1024 ** 2
        speed = mib / DURATION
        print(f"  总下载: {mib:8.2f} MiB, 平均: {speed:6.2f} MiB/s\n")
    print("测试完毕。")


if __name__ == "__main__":
    main()
