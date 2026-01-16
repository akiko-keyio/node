from __future__ import annotations

import os
import sys
import time
import logging
import subprocess
from pathlib import Path
from typing import Optional

import requests  # pip install requests

logger = logging.getLogger(__name__)


# ---------- 工具函数 ---------- #
def _remote_file_size(url: str, timeout: int = 10) -> Optional[int]:
    """
    通过 HEAD 请求获取远端文件的 Content-Length 字节数。
    若服务器不返回 Content-Length，则返回 None。
    """
    try:
        resp = requests.head(url, allow_redirects=True, timeout=timeout)
        cl = resp.headers.get("Content-Length")
        return int(cl) if cl is not None else None
    except Exception as exc:  # 包括网络错误、转换错误等
        logger.warning(f"HEAD 请求失败，无法获取 Content-Length: {exc!r}")
        return None


def _download_once(url: str, dst: Path) -> int:
    """
    调用 aria2c 下载一次；成功时返回实际文件大小（字节）。
    任何异常均直接抛出，由上层处理。
    """
    logger.info(f"Transferring {url} To {dst}")
    # 把虚拟环境的 Scripts 目录插到 PATH 最前面
    venv_scripts = os.path.dirname(sys.executable)
    os.environ["PATH"] = venv_scripts + os.pathsep + os.environ.get("PATH", "")
    cmd = [
        "aria2c", "-c",
        "--no-conf",
        "--split=16", "--max-connection-per-server=16",
        "--file-allocation=none",
        "-d", os.path.dirname(dst) or ".",
        "-o", os.path.basename(dst),
        url,
    ]

    try:
        subprocess.check_call(cmd)
    except FileNotFoundError:
        raise FileNotFoundError("aria2c not found – please install it or adjust PATH")
    except subprocess.CalledProcessError as exc:
        # 抛出运行时错误，方便外层统一处理
        raise RuntimeError(f"aria2c returned non-zero exit code {exc.returncode}") from exc

    return dst.stat().st_size

def transfer_with_idm(
        url: str,
        dst: Path,
):
    idm_engine = r'C:\Program Files (x86)\Internet Download Manager\IDMan.exe'
    subprocess.call([idm_engine, '/d', url, '/p', dst.parent, '/f', dst.name, '/a'])
    subprocess.call([idm_engine, '/s'])

# ---------- 公开函数 ---------- #
def transfer_with_retry(
        url: str,
        dst: str | Path,
        *,
        expected_size: Optional[int] = None,
        retries: int = 3,
        backoff: int = 5,
) -> int:
    """
    下载文件；若出错或大小不符则自动重试，返回最终文件大小（字节）。

    参数
    ----
    url            下载链接
    dst            保存路径（str 或 Path）
    expected_size  预期文件大小；若为 None，则先尝试 HEAD 获取
    retries        最多重试次数
    backoff        初始退避秒数，之后指数递增（*2）
    """
    dst = Path(dst).expanduser().resolve()

    if expected_size is None:
        expected_size = _remote_file_size(url)

    for attempt in range(1, retries + 1):
        try:
            size = _download_once(url, dst)

            if expected_size is not None and size != expected_size:
                raise ValueError(
                    f"Size mismatch: got {size:,d} B, expected {expected_size:,d} B"
                )

            logger.info("Download succeeded: %s (%s bytes)", dst, size)
            return size

        except Exception as exc:
            logger.warning("[%d/%d] Download failed: %s", attempt, retries, exc)

            # 清理不完整文件，准备重试
            if dst.exists():
                try:
                    dst.unlink()
                except OSError:
                    pass

            if attempt == retries:
                logger.error("All retries exhausted – giving up.")
                raise

            sleep_time = backoff * (2 ** (attempt - 1))
            logger.info("Retrying in %d s …", sleep_time)
            time.sleep(sleep_time)

    # 理论上到不了这里；加一层保险
    raise RuntimeError("Unexpected flow in transfer_with_retry")
