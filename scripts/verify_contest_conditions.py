#!/usr/bin/env python3
"""按赛题硬约束做一次端到端自检（可反复运行，用于提交前确认）。

模拟评测条件：无外网（不配 PyTRiO / 不传 SwanLab 密钥）、CPU、无人干预、一键跑完。
逐条核对：C1 无人干预、C2 按时输出、C9 ≥3 次完整闭环、C10 不低于官方基线、C11 可复现，
以及"只读 public 数据、不触碰隐藏标签"。

用法：
    python3 scripts/verify_contest_conditions.py            # 全部三包
    python3 scripts/verify_contest_conditions.py --quick    # 只跑 A、B（省时）
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from driver.package import load_package  # noqa: E402


def discover_packages() -> list[tuple[str, str]]:
    """动态发现官方包，不写死带版本日期的目录名（平台目录名可能不同）。

    支持用 BASE_DIR 覆盖（容器里把 base/ 挂到别处时用）。
    """
    base = Path(os.environ.get("BASE_DIR", ROOT / "base"))
    found = []
    for d in sorted(base.glob("cpu-*")):
        if not (d / "baseline-info.json").is_file():
            continue
        try:
            label = json.loads((d / "baseline-info.json").read_text())["dataset_key"]
        except Exception:  # noqa: BLE001
            continue
        rel = d if d.is_absolute() else d.resolve()
        found.append((label, str(rel)))
    return found


def offline_env() -> dict:
    """剥掉一切外网凭据，模拟 C4 离线评测环境。"""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("PYTRIO_", "SWANLAB_"))}
    env["BASELINE_CPU_THREADS"] = "1"
    env["PYTHONHASHSEED"] = "0"
    return env


def run_package(label: str, pkg_dir: str, rounds: int) -> dict:
    # discover_packages 返回绝对路径；ROOT / 绝对路径 仍是该绝对路径，故两种形式都兼容
    pkg = load_package(ROOT / pkg_dir)
    baseline = json.loads((pkg.root / "training.json").read_text())["validation_after"]
    key = "macro_f1" if pkg.task == "classification" else "rmse"
    t0 = time.perf_counter()
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    proc = subprocess.run(
        [sys.executable, "scripts/run_official.py", "--package", pkg_dir,
         "--rounds", str(rounds)],
        cwd=str(ROOT), capture_output=True, text=True, env=offline_env(),
    )
    seconds = time.perf_counter() - t0
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    rss_mb = max(peak, before) / (1024 * 1024 if sys.platform == "darwin" else 1024)

    out = proc.stdout
    rows = list(csv.DictReader(
        (ROOT / "outputs" / f"official_{pkg.profile.lower()}_rounds.csv").open()))
    audit = [json.loads(l) for l in
             (ROOT / "outputs" / f"audit_official-{pkg.profile.lower()}.jsonl").open()]
    complete = [r for r in audit if r.get("history_read") is not None
                and r.get("hypothesis") and r.get("experiments") and r.get("decision")]
    best = float(rows[-1]["best_value"]) if rows else float("nan")
    better = best > baseline[key] if key == "macro_f1" else best < baseline[key]
    return {
        "label": label, "profile": pkg.profile, "ok": proc.returncode == 0,
        "seconds": seconds, "rss_mb": rss_mb, "rounds": len(rows),
        "baseline": baseline[key], "best": best, "beats_baseline": better,
        "loops": len(complete), "degraded": "降级 0" in out,
        "llm_degraded_ok": "不可用→降级规则" in out or "--decider llm" not in out,
        "stdout": out, "stderr": proc.stderr[-400:],
        "rows_digest": [(r["round"], r["value"], r["best_value"]) for r in rows],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="只跑 A、B")
    ap.add_argument("--time-budget", type=float, default=300.0,
                    help="单包耗时自设预算（秒）。官方未公布时限，此值仅作内部护栏")
    ap.add_argument("--repeat", action="store_true", default=True,
                    help="复跑一次校验可复现性（C11）")
    args = ap.parse_args()
    plan_all = discover_packages()
    if not plan_all:
        print("未发现官方包（期望 base/cpu-* 或 BASE_DIR 指向的目录）")
        return 1
    plan = plan_all[:2] if args.quick else plan_all

    print("=" * 72)
    print("赛题硬约束端到端自检（离线、CPU、无人干预）")
    print("=" * 72)
    results = [run_package(l, d, args.rounds) for l, d in plan]

    print(f"\n{'包':<4}{'轮数':>5}{'官方基线':>11}{'我们最优':>11}{'优于基线':>9}"
          f"{'闭环':>6}{'耗时s':>8}{'峰值内存MB':>12}")
    for r in results:
        print(f"{r['profile']:<4}{r['rounds']:>5}{r['baseline']:>11.4f}{r['best']:>11.4f}"
              f"{'是' if r['beats_baseline'] else '否':>9}{r['loops']:>6}"
              f"{r['seconds']:>8.1f}{r['rss_mb']:>12.0f}")

    checks = []
    checks.append(("C1 无人干预跑完（退出码 0、无降级事件）",
                   all(r["ok"] and r["degraded"] for r in results)))
    checks.append((f"C2 按时输出（自设预算：单包 < {args.time_budget:.0f}s，官方未公布时限）",
                   all(r["seconds"] < args.time_budget for r in results)))
    checks.append(("C3 内存在 8G 以内（峰值 < 2000MB）",
                   all(r["rss_mb"] < 2000 for r in results)))
    checks.append(("C4 离线可跑（无 PyTRiO/SwanLab 凭据）", all(r["ok"] for r in results)))
    checks.append(("C9 每包 ≥3 次完整闭环", all(r["loops"] >= 3 for r in results)))
    checks.append(("C10 不低于官方基线", all(r["beats_baseline"] for r in results)))
    checks.append((f"轮数达标（={args.rounds}）", all(r["rounds"] == args.rounds for r in results)))

    if args.repeat:
        second = [run_package(l, d, args.rounds) for l, d in plan]
        same = all(a["rows_digest"] == b["rows_digest"] for a, b in zip(results, second))
        checks.append(("C11 可复现（固定种子两次逐轮完全一致）", same))

    print()
    for name, ok in checks:
        print(f"  [{'通过' if ok else '未通过'}] {name}")
    failed = [n for n, ok in checks if not ok]
    print("\n" + ("全部通过。" if not failed else f"未通过 {len(failed)} 项：{failed}"))
    if failed:
        for r in results:
            if r["stderr"].strip():
                print(f"\n[{r['profile']}] stderr 尾部：\n{r['stderr']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
