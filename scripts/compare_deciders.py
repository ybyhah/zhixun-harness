"""三条决策臂（不训/规则/27B 慢通路）在官方包逐轮续训流上的对照评测。

用法：
    python3 scripts/compare_deciders.py --packages a,b,c --arms notrain,rule,llm --rounds 5
产物：outputs/COMPARISON_deciders.md、outputs/compare_deciders.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from driver.loop import OfficialDriver, _baseline_value  # noqa: E402
from driver.metric import to_score  # noqa: E402
from driver.package import load_package  # noqa: E402
from driver.policy import AdaptivePolicy, RulePolicy  # noqa: E402
from driver.llm import HybridDecider, LLMClient, LLMDecider  # noqa: E402
from harness import metrics  # noqa: E402

PKG_DIRS = {
    "a": "base/cpu-a-guide-20261110-v1",
    "b": "base/cpu-b-rul-20261110-v1",
    "c": "base/cpu-c-guide-20261110-v1",
}


def _sum_tokens(audit_path: Path) -> tuple[int, int]:
    """从 audit.jsonl 累加 token 与降级轮数。"""
    tok = deg = 0
    if not audit_path.exists():
        return tok, deg
    for line in audit_path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        tok += int(d.get("cost", {}).get("tokens", 0) or 0)
        if d.get("decision", {}).get("llm_degraded"):
            deg += 1
    return tok, deg


def _make_policy(arm: str, epochs: int, base_lr: float, pkg):
    if arm == "rule":
        return RulePolicy(base_epochs=epochs, base_lr=base_lr)
    if arm == "adaptive":
        return AdaptivePolicy(base_epochs=epochs, base_lr=base_lr)
    if arm in ("llm", "hybrid"):
        llm = LLMDecider(
            profile=pkg.profile, task=pkg.task, metric_name=pkg.metric_name,
            higher_better=pkg.higher_better, baseline_value=0.0,
            fallback=RulePolicy(base_epochs=epochs, base_lr=base_lr),
            client=LLMClient(),
        )
        if arm == "llm":
            return llm
        # 混合臂的快通路用 AdaptivePolicy：被采纳后保持超参，LLM 找到的好配置不会被丢掉
        return HybridDecider(AdaptivePolicy(base_epochs=epochs, base_lr=base_lr), llm)
    raise ValueError(f"未知决策臂：{arm}")


def run_arm(pkg, name, arm, rounds, epochs, candidates, base_value, base_score):
    out = ROOT / "outputs" / "compare" / name / arm
    if arm == "notrain":
        scores = dep_scores = [base_score] * rounds
        sec, tok, deg, prom, tr, delivered, calls = 0.0, 0, 0, 0, 0, base_value, 0
    else:
        policy = _make_policy(arm, epochs, base_lr=_lr_of(pkg), pkg=pkg)
        drv = OfficialDriver(pkg, rounds_dir=out / "rounds", rounds=rounds,
                             policy=policy, candidates=candidates,
                             gate_mode="significance", audit_path=out / "audit.jsonl")
        if arm in ("llm", "hybrid"):
            policy.baseline_value = drv.baseline_value
        res = drv.run()
        # 候选轨迹：每轮实际训练出的 value（含被门控拒绝的探索）
        scores = [to_score(pkg, r.value) for r in res.rows]
        # 部署轨迹：每轮实际在线的 best_value（拒绝的实验不影响部署），更贴近平台口径
        dep_scores = [to_score(pkg, r.best_value) for r in res.rows]
        sec, prom, tr = res.seconds, res.significant_promotions, res.trainings
        tok, deg = _sum_tokens(out / "audit.jsonl")
        delivered, calls = res.deliverable_value, res.llm_calls
    s = metrics.summarize(scores, baseline_composite=base_score,
                          trainings=tr, seconds=sec, promotions=prom)
    sdep = metrics.summarize(dep_scores, baseline_composite=base_score,
                             trainings=tr, seconds=sec, promotions=prom)
    return {
        "pkg": name, "arm": arm,
        "gain": s.gain_vs_baseline, "composite": s.composite, "worst_stage": s.worst_stage,
        "dep_gain": sdep.gain_vs_baseline, "dep_composite": sdep.composite,
        "dep_worst": sdep.worst_stage, "recovery": s.recovery_mean,
        "sig_promos": prom, "trainings": tr, "seconds": sec,
        "tokens": tok, "llm_calls": calls, "degraded": deg, "delivered": delivered,
    }



def _lr_of(pkg) -> float:
    return float(json.loads(pkg.config_path().read_text()).get("learning_rate", 0.003))


def write_report(rows, md_path: Path, csv_path: Path):
    cols = ["pkg", "arm", "dep_gain", "gain", "dep_composite", "composite",
            "worst_stage", "recovery", "sig_promos", "trainings", "seconds",
            "tokens", "llm_calls", "degraded", "delivered"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])
    lines = [
        "# 决策臂对照评测（notrain / rule / adaptive / llm / hybrid）",
        "",
        "- 口径：全部用统一 score（分类 Macro-F1，回归 -RMSE；越大越好）。",
        "- **部署增益**：用每轮在线的 best_value 算，被门控拒绝的探索不计入 —— 最贴近平台「部署模型 vs 不训」口径。",
        "- 候选增益：用每轮训练出的 value 算，含被拒的探索尝试（反映探索波动）。",
        "- 恢复：跌破历史最佳-tol 后回升所需批次均值（越小越好）。",
        "- hybrid：常态走 adaptive 快通路；连续 2 轮没上分，或近 2 轮提升不足基线 0.5%（爬坡放缓）时唤醒 27B。",
        "",
        "| 包 | 臂 | 部署增益 | 候选增益 | 部署综合 | 候选综合 | 最差阶段 | 恢复 | 显著上线 | 训练次 | 耗时s | tokens | LLM调用 | 降级轮 | 交付值 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            "| {pkg} | {arm} | {dep_gain:+.4f} | {gain:+.4f} | {dep_composite:.4f} "
            "| {composite:.4f} | {worst_stage:.4f} | {recovery:.2f} | {sig_promos} "
            "| {trainings} | {seconds:.1f} | {tokens} | {llm_calls} | {degraded} "
            "| {delivered:.4f} |".format(**r))
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packages", default="a,b,c")
    ap.add_argument("--arms", default="notrain,rule,llm")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--candidates", type=int, default=1)
    args = ap.parse_args()

    pkgs = [p.strip().lower() for p in args.packages.split(",") if p.strip()]
    arms = [a.strip().lower() for a in args.arms.split(",") if a.strip()]
    rows = []
    for name in pkgs:
        pkg = load_package(ROOT / PKG_DIRS[name])
        base = _baseline_value(pkg)
        base_score = to_score(pkg, base)
        print(f"==== 包 {name}  基线值={base:.4f} (score={base_score:.4f}) ====")
        for arm in arms:
            print(f"  [{name}/{arm}] 运行中 ...")
            r = run_arm(pkg, name, arm, args.rounds, args.epochs,
                        args.candidates, base, base_score)
            rows.append(r)
            print(f"    增益={r['gain']:+.4f} 交付={r['delivered']:.4f} "
                  f"耗时={r['seconds']:.1f}s tokens={r['tokens']} 降级={r['degraded']}")
    out_dir = ROOT / "outputs"
    write_report(rows, out_dir / "COMPARISON_deciders.md",
                 out_dir / "compare_deciders.csv")
    print(f"\n对照报告写入 {out_dir / 'COMPARISON_deciders.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


