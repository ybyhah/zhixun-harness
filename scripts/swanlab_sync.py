#!/usr/bin/env python3
"""把离线 SwanLab 记录上传到云端，并提示设为公开（C8 硬性要求）。

评测环境无外网（C4），SwanLab 以 offline 模式把记录落在本地 `swanlog/`；
赛后在有网环境运行本脚本上传，再到网页端把实验设为公开，否则官方访问不到。

用法：
    # 先登录（一次）：swanlab login  （粘贴 API Key）
    python3 scripts/swanlab_sync.py                 # 上传 swanlog/ 下全部离线 run
    python3 scripts/swanlab_sync.py --dir swanlog   # 指定目录
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "outputs" / "swanlog"),
                    help="离线记录目录（默认 outputs/swanlog/）")
    args = ap.parse_args()

    base = Path(args.dir)
    if not base.exists():
        print(f"未找到离线记录目录：{base}")
        print("请先用 `--swanlab`（offline 模式）跑一次实验生成 swanlog/")
        return 1

    runs = sorted(p for p in base.glob("run-*") if p.is_dir())
    if not runs:
        print(f"{base} 下没有离线 run")
        return 1

    # swanlab 可执行文件不一定在 PATH 上（pip --user / framework Python），回退到 python -m swanlab
    cli = ["swanlab"] if __import__("shutil").which("swanlab") else [sys.executable, "-m", "swanlab"]
    try:
        subprocess.run(cli + ["--help"], capture_output=True, check=True)
    except Exception:  # noqa: BLE001
        print("未找到 swanlab，请先 `pip install swanlab` 并 `swanlab login`")
        return 1

    print(f"准备上传 {len(runs)} 个离线 run ...")
    failed = []
    for run in runs:
        print(f"  swanlab sync {run.name}")
        r = subprocess.run(cli + ["sync", str(run)], capture_output=True, text=True)
        if r.returncode != 0:
            failed.append(run.name)
            print(f"    失败：{r.stderr.strip().splitlines()[-1:] or r.stdout[-200:]}")

    print("\n上传完成。" if not failed else f"\n完成，{len(failed)} 个失败：{failed}")
    print("=" * 60)
    print("提示：实验用 public=True 创建，新项目会直接公开；若是之前已存在的私有项目，")
    print("  仍需到网页端 项目页 → Settings → Visibility → Public（C8）。")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
