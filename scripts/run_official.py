"""驱动官方 base/ 包续训的入口（评分对象 = 官方 CPU MLP 续训检查点）。

用法：
    python3 scripts/run_official.py --package base/cpu-a-guide-20261110-v1 --rounds 10
    python3 scripts/run_official.py --package base/cpu-b-rul-20261110-v1 --rounds 10 --swanlab
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from driver.loop import OfficialDriver, load_verdicts  # noqa: E402
from driver.metric import summarize_rounds  # noqa: E402
from driver.package import load_package  # noqa: E402
from driver.policy import AdaptivePolicy, RulePolicy  # noqa: E402
from driver.llm import HybridDecider, LLMClient, LLMDecider  # noqa: E402
from harness.record.swan import SwanRecorder  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True, help="base/ 下的包目录")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=5, help="每轮续训 epoch 数")
    ap.add_argument("--min-gain", type=float, default=0.0)
    ap.add_argument("--decider", choices=["rule", "llm", "hybrid"], default="rule",
                    help="llm=PyTRiO 慢通路（无端点自动降级规则）；"
                         "hybrid=常态走规则，停滞时才唤醒 LLM")
    ap.add_argument("--patience", type=int, default=2,
                    help="hybrid：连续多少轮没上分就唤醒 LLM")
    ap.add_argument("--slow-gain", type=float, default=0.005,
                    help="hybrid：近 2 轮已上线最优提升不足基线的该比例（爬坡放缓）也唤醒 LLM；0=关闭")
    ap.add_argument("--llm-patience", type=int, default=2,
                    help="hybrid：LLM 连续多少次提案未被采纳就熔断（有新上线后恢复）；0=关闭")
    ap.add_argument("--candidates", type=int, default=1,
                    help="每轮候选超参数（>1 时每轮多试几个，验证择优）")
    ap.add_argument("--policy", choices=["rule", "adaptive"], default="rule",
                    help="adaptive=停滞检测+换轴+无效轮跳过（省算力）")
    ap.add_argument("--gate", choices=["plain", "significance"], default="significance",
                    help="significance(默认)=配对 bootstrap 显著性门控；plain=纯数值比较")
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--p-threshold", type=float, default=0.95)
    ap.add_argument("--verdicts", default="", help="平台侧裁决 JSON，如 {\"round-3\":\"rejected\"}")
    ap.add_argument("--swanlab", action="store_true")
    ap.add_argument("--swanlab-mode", choices=["offline", "cloud"], default="offline",
                    help="offline=落本地 swanlog（评测离线用）；cloud=直接上传（需 SWANLAB_API_KEY）")
    ap.add_argument("--swanlab-project", default="zhixun-harness-official")
    args = ap.parse_args()

    pkg = load_package(ROOT / args.package if not Path(args.package).is_absolute()
                       else args.package)
    tag = f"official-{pkg.profile.lower()}"
    base_lr = float(json.loads(pkg.config_path().read_text()).get("learning_rate", 0.003))
    swan = SwanRecorder(
        enabled=args.swanlab, project=args.swanlab_project,
        mode=args.swanlab_mode,
        # 名字带决策器，平台上能区分 rule / llm / hybrid 的实验
        run_name=f"{tag}-{args.decider}",
        config={"profile": pkg.profile, "task": pkg.task,
                "parameter_count": pkg.parameter_count, "rounds": args.rounds,
                # hybrid 的快通路固定为 adaptive（见下方构造），记录实际生效的策略
                "decider": args.decider,
                "policy": "adaptive" if args.decider == "hybrid" else args.policy,
                "gate": args.gate,
                "candidates": args.candidates, "base_epochs": args.epochs,
                "base_learning_rate": base_lr,
                **({"llm_model": os.environ.get("PYTRIO_MODEL", ""),
                    "patience": args.patience, "slow_gain": args.slow_gain,
                    "llm_patience": args.llm_patience}
                   if args.decider in ("llm", "hybrid") else {})},
        # 落在 outputs/ 下：Docker 里 outputs 是挂载卷，--rm 后记录不丢
        log_dir=str(ROOT / "outputs" / "swanlog"),
    )
    rule = RulePolicy(base_epochs=args.epochs, base_lr=base_lr)
    if args.policy == "adaptive":
        rule = AdaptivePolicy(base_epochs=args.epochs, base_lr=base_lr)
    if args.decider in ("llm", "hybrid"):
        llm = LLMDecider(
            profile=pkg.profile, task=pkg.task, metric_name=pkg.metric_name,
            higher_better=pkg.higher_better, baseline_value=0.0,
            # hybrid 下快通路与 LLM 兜底各用一个实例，避免共享去重状态
            fallback=(rule if args.decider == "llm"
                      else RulePolicy(base_epochs=args.epochs, base_lr=base_lr)),
            client=LLMClient(),
        )
        policy = llm if args.decider == "llm" else HybridDecider(
            # 混合的快通路固定用 AdaptivePolicy：被采纳后保持超参，LLM 找到的好配置不会被丢回默认
            AdaptivePolicy(base_epochs=args.epochs, base_lr=base_lr), llm,
            patience=args.patience, min_rel_gain=args.slow_gain,
            llm_patience=args.llm_patience)
        mode = ("LLM 慢通路" if args.decider == "llm"
                else f"快慢混合（停滞 {args.patience} 轮或爬坡放缓 <{args.slow_gain:.1%} 唤醒 LLM）")
        print(f"决策器：{mode}（PyTRiO {'可用' if llm.client.available else '不可用→降级规则'}）")
    else:
        policy = rule
    driver = OfficialDriver(
        pkg, rounds_dir=ROOT / "outputs" / "official" / pkg.profile.lower(),
        rounds=args.rounds, min_gain=args.min_gain,
        policy=policy, candidates=args.candidates,
        gate_mode=args.gate, bootstrap=args.bootstrap,
        p_threshold=args.p_threshold, verdicts=load_verdicts(args.verdicts or None),
        audit_path=ROOT / "outputs" / f"audit_{tag}.jsonl", swan=swan,
    )
    if args.decider in ("llm", "hybrid"):
        policy.baseline_value = driver.baseline_value
    print(f"官方包 {pkg.root.name}  profile={pkg.profile}  task={pkg.task}  "
          f"params={pkg.parameter_count}")
    print(f"官方基线验证 {pkg.metric_name} = {driver.baseline_value:.4f}")
    res = driver.run()

    metric = pkg.metric_name
    print(f"\n{'轮':<5}{metric:>12}{'最优':>12}{'判定':>10}{'耗时s':>8}")
    for r in res.rows:
        print(f"{r.round:<5}{r.value:>12.4f}{r.best_value:>12.4f}"
              f"{r.action:>10}{r.seconds:>8.1f}")
    summary = summarize_rounds(pkg, [r.value for r in res.rows],
                               seconds=res.seconds, trainings=res.trainings,
                               promotions=res.promotions)
    improve = (res.best_value - res.baseline_value) if pkg.higher_better \
        else (res.baseline_value - res.best_value)
    print(f"\n基线 {metric} {res.baseline_value:.4f} → 最优 {res.best_value:.4f}"
          f"（{'提升' if improve >= 0 else '退化'} {abs(improve):.4f}）")
    print(f"上线 {res.promotions} 次 / 续训 {res.trainings} 次 / 跳过 {res.skipped} 次 / 降级 "
          f"{len(res.failures)} / 用时 {res.seconds:.1f}s")
    if args.decider in ("llm", "hybrid"):
        n, calls = res.llm_degraded_rounds, res.llm_calls
        warn = ("  ← 全部降级：LLM 未生效，检查端点/网络/证书"
                if calls and n >= calls else "")
        print(f"LLM 慢通路：调用 {calls}/{len(res.rows)} 轮，其中降级 {n} 轮{warn}")
    if args.gate == "significance":
        d = res.deliverable_value
        print(f"显著性链：上线 {res.significant_promotions} 次；"
              f"最终交付 {metric}={d:.4f}"
              f"（{'显著性确认' if res.best_significant_checkpoint else '回退验证最优'}）")
        print(f"交付 checkpoint：{res.deliverable_checkpoint}")
    if res.rejected_by_verdict:
        print(f"被平台裁决拒绝的轮次：{res.rejected_by_verdict}")
    print(f"完整闭环记录 {len(driver.audit.complete_loops())} 条（门槛 ≥ 3）")
    if swan.enabled:
        if swan.degraded:
            print(f"SwanLab 降级为本地记录：{swan.degrade_reason}")
        elif args.swanlab_mode == "cloud":
            print("SwanLab：已上传云端（记得在网页端把项目设为公开，C8 要求）")
        else:
            print("SwanLab：offline 已写入 outputs/swanlog/，可用 scripts/swanlab_sync.py 事后上传")

    out = ROOT / "outputs" / f"official_{pkg.profile.lower()}_rounds.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["round", "value", "best_value", "accepted", "seconds"])
        for r in res.rows:
            w.writerow([r.round, r.value, r.best_value, int(r.accepted), f"{r.seconds:.2f}"])
    print(f"逐轮结果写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
