"""按自定义训练损失跑满 N 轮续训，与官方损失结果对照。

只改训练损失（运行时注入 WeightedModel），官方 train.py 的校验与报告、
predict 协议、checkpoint 格式都不变。产物写 outputs/loss/，不动 base/。

用法：
    python3 scripts/run_loss_variant.py --package base/cpu-c-guide-20261110-v1 --loss asym --alpha 2
    python3 scripts/run_loss_variant.py --package base/cpu-a-guide-20261110-v1 --loss weighted_ce
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from driver.loop import OfficialDriver  # noqa: E402
from driver.metric import nasa_asym_score, over_estimate_rate  # noqa: E402
from driver.migrate import _load_official_model_cls  # noqa: E402
from driver.package import load_package  # noqa: E402
from driver.policy import RulePolicy  # noqa: E402
from driver.weighted import write_launcher  # noqa: E402
from harness.record.swan import SwanRecorder  # noqa: E402


def regression_extras(pkg, checkpoint: Path) -> tuple[float, float]:
    """回归任务在验证集上的（NASA 非对称分, 高估比例）。

    NASA 非对称分是 C-MAPSS（循环）专用口径，`exp(d/10)` 对 B 的分钟量级会溢出成
    天文数字，没有意义，因此只对 profile C 计算，其余返回 NaN。
    """
    Model = _load_official_model_cls(pkg.root)
    sys.path.insert(0, str(pkg.root))
    from baselines.data import read_dataset

    model = Model.load(checkpoint)
    va = read_dataset(pkg.validation_jsonl, pkg.task, [], model.meta["paths"])
    pred = model.predict(va["x"])
    asym = nasa_asym_score(va["y"], pred) if pkg.profile == "C" else float("nan")
    return asym, over_estimate_rate(va["y"], pred)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--loss", default="official",
                    choices=["official", "weighted_ce", "asym", "huber"])
    ap.add_argument("--alpha", type=float, default=1.0, help="asym 的高估惩罚倍数")
    ap.add_argument("--delta", type=float, default=1.0, help="huber 的 δ")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--swanlab", action="store_true")
    args = ap.parse_args()

    pkg_dir = Path(args.package)
    pkg = load_package(ROOT / pkg_dir if not pkg_dir.is_absolute() else pkg_dir)
    variant = args.loss if args.loss != "asym" else f"asym-a{args.alpha:g}"
    if args.loss == "huber":
        variant = f"huber-d{args.delta:g}"
    work = ROOT / "outputs" / "loss" / pkg.profile.lower() / variant
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    launcher = write_launcher(work / "patched_train.py", pkg.root, ROOT / "src",
                              args.loss, args.alpha, args.delta)
    cfg = json.loads(pkg.config_path().read_text())
    tag = f"loss-{pkg.profile.lower()}-{variant}"
    swan = SwanRecorder(enabled=args.swanlab, project="zhixun-harness-loss",
                        mode="offline", run_name=tag,
                        config={"profile": pkg.profile, "loss": args.loss,
                                "alpha": args.alpha, "delta": args.delta})
    driver = OfficialDriver(
        pkg, rounds_dir=work / "rounds", rounds=args.rounds,
        policy=RulePolicy(base_epochs=args.epochs,
                          base_lr=float(cfg.get("learning_rate", 0.003))),
        audit_path=ROOT / "outputs" / f"audit_{tag}.jsonl", swan=swan,
        entry_override=launcher,
    )
    print(f"包 {pkg.root.name} | 损失 {variant} | 官方基线 {pkg.metric_name} "
          f"= {driver.baseline_value:.4f}")
    res = driver.run()

    print(f"\n{'轮':<5}{pkg.metric_name:>12}{'最优':>12}{'判定':>10}{'耗时s':>8}")
    for r in res.rows:
        print(f"{r.round:<5}{r.value:>12.4f}{r.best_value:>12.4f}{r.action:>10}{r.seconds:>8.1f}")
    improve = (res.best_value - res.baseline_value) if pkg.higher_better \
        else (res.baseline_value - res.best_value)
    print(f"\n损失 {variant}：基线 {res.baseline_value:.4f} → 最优 {res.best_value:.4f}"
          f"（{'提升' if improve >= 0 else '退化'} {abs(improve):.4f}）")
    if pkg.task == "regression" and res.best_checkpoint:
        asym, over = regression_extras(pkg, res.best_checkpoint)
        extra = f"NASA 非对称分 {asym:.1f}（越低越好）| " if asym == asym else ""
        print(f"{extra}高估比例 {over:.3f}")
    print(f"上线 {res.promotions} / 续训 {res.trainings} / 降级 {len(res.failures)} / "
          f"用时 {res.seconds:.1f}s\n产物 {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
