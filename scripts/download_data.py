"""下载 UCI Gas Sensor Array Drift 数据集。

数据集：16 个金属氧化物传感器，6 类气体，128 维特征，10 个批次，跨 36 个月。
文件格式为 libsvm 风格：  <label> <idx>:<value> ...
"""

from __future__ import annotations

import io
import shutil
import ssl
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
EXPECTED = [f"batch{i}.dat" for i in range(1, 11)]

# 公开资料记载的各批次样本总数，用于校验下载完整性
EXPECTED_ROWS = {
    1: 445, 2: 1244, 3: 1586, 4: 161, 5: 197,
    6: 2300, 7: 3613, 8: 294, 9: 470, 10: 3600,
}


def _fetch(url: str) -> bytes:
    """优先用 certifi 的证书，失败则退回 curl。

    部分 Python 安装缺少系统根证书，直接 urlopen 会报
    CERTIFICATE_VERIFY_FAILED，因此需要这个兜底。
    """
    try:
        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, timeout=180, context=ctx) as resp:
            return resp.read()
    except Exception as exc:  # noqa: BLE001 - 任何失败都退回 curl
        print(f"  urllib 失败({type(exc).__name__})，改用 curl")
        if shutil.which("curl") is None:
            raise
        out = subprocess.run(
            ["curl", "-sSL", "--max-time", "180", url],
            capture_output=True,
            check=True,
        )
        return out.stdout


def download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if all((RAW_DIR / n).exists() for n in EXPECTED):
        print(f"数据已存在于 {RAW_DIR}，跳过下载")
        return
    print(f"下载 {URL}")
    payload = _fetch(URL)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for member in zf.namelist():
            name = Path(member).name
            if name in EXPECTED:
                (RAW_DIR / name).write_bytes(zf.read(member))
                print(f"  写入 {name}")


def verify() -> bool:
    ok = True
    for i, name in enumerate(EXPECTED, start=1):
        path = RAW_DIR / name
        if not path.exists():
            print(f"[缺失] {name}")
            ok = False
            continue
        rows = sum(1 for line in path.read_text().splitlines() if line.strip())
        flag = "OK" if rows == EXPECTED_ROWS[i] else f"期望 {EXPECTED_ROWS[i]}"
        if rows != EXPECTED_ROWS[i]:
            ok = False
        print(f"  {name:12s} {rows:5d} 行  {flag}")
    return ok


if __name__ == "__main__":
    download()
    sys.exit(0 if verify() else 1)
