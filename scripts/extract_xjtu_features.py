"""A2 XJTU-SY 特征提取：RAR 分卷 → data/raw_xjtu/features.npz（一次性，仅本地开发用）。

全量解压约 11GB，超过本机剩余空间，因此逐卷、逐轴承解到临时目录，
提完特征立即删除。跨卷边界的文件单卷解不完整，按声明大小校验后记为缺失。

用法：
    python3 scripts/extract_xjtu_features.py
    python3 scripts/extract_xjtu_features.py --bearings 1_5,2_4 --out /tmp/f.npz
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harness.datasets.vibration import FEATURE_NAMES, extract_features  # noqa: E402

CONDITIONS = {"35Hz12kN": 0, "37.5Hz11kN": 1, "40Hz10kN": 2}
# 官方寿命表（分钟），见 XJTU-SY 数据集说明
LIFE = {
    "1_1": 123, "1_2": 161, "1_3": 158, "1_4": 122, "1_5": 52,
    "2_1": 491, "2_2": 161, "2_3": 533, "2_4": 42, "2_5": 339,
    "3_1": 2538, "3_2": 2496, "3_3": 371, "3_4": 1515, "3_5": 114,
}
ENTRY = re.compile(r"/(?P<cond>[^/]+)/Bearing(?P<b>\d_\d)/(?P<m>\d+)\.csv$")
MIN_FREE_BYTES = 3 * 1024 ** 3


def list_volume(vol: Path) -> dict[str, dict[str, int]]:
    """返回 {bearing: {归档内路径: 声明字节数}}。"""
    out = subprocess.run(["bsdtar", "-tvf", str(vol)], capture_output=True, text=True)
    groups: dict[str, dict[str, int]] = defaultdict(dict)
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) < 9:
            continue
        name, size = parts[-1], parts[4]
        m = ENTRY.search(name)
        if m and size.isdigit():
            groups[m["b"]][name] = int(size)
    return groups


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rar-dir", default=str(ROOT / "data/raw_xjtu/Data"))
    ap.add_argument("--out", default=str(ROOT / "data/raw_xjtu/features.npz"))
    ap.add_argument("--bearings", default="", help="逗号分隔，如 1_5,2_4；默认全部")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"缓存已存在：{out}（加 --force 重新提取）")
        return 0
    if shutil.which("bsdtar") is None:
        print("未找到 bsdtar，无法读取 RAR 分卷")
        return 1
    vols = sorted(Path(args.rar_dir).glob("*.part*.rar"))
    if not vols:
        print(f"{args.rar_dir} 下没有 RAR 分卷")
        return 1
    wanted = {b.strip() for b in args.bearings.split(",") if b.strip()}

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="xjtu_", dir=out.parent))
    rows, bearing, cond, minute, life, missing = [], [], [], [], [], []
    stats: dict[str, list[float]] = defaultdict(lambda: [0, 0, 0.0])  # 样本, 缺失, 秒
    try:
        for vol in vols:
            for b, entries in sorted(list_volume(vol).items()):
                if wanted and b not in wanted:
                    continue
                if shutil.disk_usage(out.parent).free < MIN_FREE_BYTES:
                    raise SystemExit("磁盘剩余空间不足 3GB，停止提取")
                t0 = time.perf_counter()
                work = tmp_root / f"{vol.stem}_{b}"
                work.mkdir()
                # 卷尾的跨卷文件会让 bsdtar 返回非零，靠下面的大小校验识别
                subprocess.run(["bsdtar", "-xf", str(vol), "-C", str(work), *entries],
                               capture_output=True)
                for name, size in entries.items():
                    p = work / name
                    if not p.exists() or p.stat().st_size != size:
                        missing.append(name)
                        stats[b][1] += 1
                        continue
                    try:
                        sig = pd.read_csv(p).to_numpy(dtype=np.float64)
                    except Exception:  # noqa: BLE001
                        missing.append(name)
                        stats[b][1] += 1
                        continue
                    m = ENTRY.search(name)
                    rows.append(extract_features(sig).astype(np.float32))
                    bearing.append(b)
                    cond.append(CONDITIONS[m["cond"]])
                    minute.append(int(m["m"]))
                    life.append(LIFE.get(b, 0))
                    stats[b][0] += 1
                shutil.rmtree(work)
                stats[b][2] += time.perf_counter() - t0
                print(f"  {vol.name} Bearing{b}: {len(entries)} 个文件")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    if not rows:
        print("没有提取到任何样本")
        return 1
    life_arr = np.asarray(life)
    minute_arr = np.asarray(minute)
    bearing_arr = np.asarray(bearing)
    for b in np.unique(bearing_arr):   # 官方表缺项时用实际最大分钟
        m = bearing_arr == b
        life_arr[m] = max(int(life_arr[m].max()), int(minute_arr[m].max()))
    np.savez_compressed(
        out, X=np.vstack(rows), feature_names=np.asarray(FEATURE_NAMES),
        bearing=bearing_arr, condition=np.asarray(cond), minute=minute_arr,
        life=life_arr, missing=np.asarray(missing, dtype=str),
    )
    print(f"\n{'轴承':<8}{'样本':>8}{'缺失':>6}{'耗时s':>8}")
    for b in sorted(stats):
        n, miss, sec = stats[b]
        print(f"{b:<8}{int(n):>8}{int(miss):>6}{sec:>8.1f}")
    print(f"共 {len(rows)} 条，缺失 {len(missing)} 个文件，已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
