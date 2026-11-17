"""加宽 + 权重迁移后跑满 N 轮续训，与未加宽结果做公平对照。

流程：函数保持式加宽（自证迁移正确）→ 生成只改 hidden 的 profile JSON →
用现有驱动器跑满轮次（门控基线仍为官方随包基线值）。产物写在 outputs/widen/，不动 base/。

用法：
    python3 scripts/widen_and_run.py --package base/cpu-a-guide-20261110-v1 --hidden 64
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from driver.loop import OfficialDriver  # noqa: E402
from driver.migrate import assert_function_preserved, widen_checkpoint  # noqa: E402
from driver.package import load_package  # noqa: E402
from driver.policy import RulePolicy  # noqa: E402
from harness.record.swan import SwanRecorder  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--hidden", type=int, required=True, help="加宽后的隐藏层宽度")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--candidates", type=int, default=1)
    ap.add_argument("--swanlab", action="store_true")
    args = ap.parse_args()

    pkg_dir = Path(args.package)
    pkg = load_package(ROOT / pkg_dir if not pkg_dir.is_absolute() else pkg_dir)
    work = ROOT / "outputs" / "widen" / pkg.profile.lower() / f"hidden-{args.hidden}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    # 1. 函数保持式加宽 + 自证
    wide_ckpt = work / "checkpoint.npz"
    count, Model = widen_checkpoint(pkg.root, pkg.checkpoint, wide_ckpt, args.hidden)
    old_meta = Model.load(pkg.checkpoint).meta
    rng = np.random.default_rng(0)
    probe = rng.normal(size=(32, len(old_meta["paths"])))
    delta = assert_function_preserved(pkg.root, pkg.checkpoint, wide_ckpt, probe)
    print(f"加宽 hidden {old_meta['hidden']} → {args.hidden}："
          f"参数量 {old_meta['parameter_count']} → {count}，"
          f"函数保持自证通过（最大偏差 {delta:.2e}）")

    # 2. 只改 hidden 的 profile（其余必须与官方一致，否则 train.py 会拒绝）
    cfg = json.loads(pkg.config_path().read_text())
    cfg["hidden"] = args.hidden
    profile_path = work / f"{pkg.profile}-hidden{args.hidden}.json"
    profile_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))

    # 3. 跑满轮次（门控基线仍是官方随包基线值）
    tag = f"widen-{pkg.profile.lower()}-h{args.hidden}"
    swan = SwanRecorder(enabled=args.swanlab, project="zhixun-harness-widen",
                        mode="offline", run_name=tag,
                        config={"profile": pkg.profile, "hidden": args.hidden,
                                "parameter_count": count})
    base_lr = float(cfg.get("learning_rate", 0.003))
    driver = OfficialDriver(
        pkg, rounds_dir=work / "rounds", rounds=args.rounds, candidates=args.candidates,
        policy=RulePolicy(base_epochs=args.epochs, base_lr=base_lr),
        audit_path=ROOT / "outputs" / f"audit_{tag}.jsonl", swan=swan,
        start_checkpoint=wide_ckpt, profile_override=profile_path,
    )
    print(f"官方基线 {pkg.metric_name} = {driver.baseline_value:.4f}（门控基线不变）")
    res = driver.run()

    print(f"\n{'轮':<5}{pkg.metric_name:>12}{'最优':>12}{'判定':>10}{'耗时s':>8}")
    for r in res.rows:
        print(f"{r.round:<5}{r.value:>12.4f}{r.best_value:>12.4f}{r.action:>10}{r.seconds:>8.1f}")
    improve = (res.best_value - res.baseline_value) if pkg.higher_better \
        else (res.baseline_value - res.best_value)
    print(f"\nhidden={args.hidden}（params {count}）：基线 {res.baseline_value:.4f} → "
          f"最优 {res.best_value:.4f}（{'提升' if improve >= 0 else '退化'} {abs(improve):.4f}）")
    print(f"上线 {res.promotions} / 续训 {res.trainings} / 降级 {len(res.failures)} / "
          f"用时 {res.seconds:.1f}s")
    print(f"产物目录 {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
