#!/usr/bin/env python3
"""用 split-half 协议裁决：hybrid（停滞/爬坡放缓时唤醒 27B）相对 rule 的提升是否真实。

背景：对照实验里 hybrid 在 B 交付 652.81（rule 666.88）、C 交付 51.18（rule 51.38）。
但这些数都是在同一份公开验证集上一边选一边评出来的，和批大小那次一样可能是选择性过拟合
（见 outputs/CONCLUSIONS.md「批大小搜索的表观增益」一节）。

做法：验证集对半分（分类按类别分层，回归随机）
  - 选择集：闭环里「是否采纳本轮」以及喂给 LLM 的历史，只用它上面的指标
  - 留出集：不参与任何选择，最后比较两种决策器各自产出的最终模型
比较的是 procedure(决策器)，不是挑出来的那个 checkpoint。多个 split 种子看方向是否一致。

用法：
    set -a; source .env; set +a
    python3 scripts/adjudicate_deciders.py --packages b,c --seeds 0 1 2
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

from driver.llm import HybridDecider, LLMClient, LLMDecider  # noqa: E402
from driver.migrate import _load_official_model_cls  # noqa: E402
from driver.official_metric import score  # noqa: E402
from driver.package import load_package  # noqa: E402
from driver.policy import AdaptivePolicy, RulePolicy  # noqa: E402
from driver.significance import paired_bootstrap  # noqa: E402
from driver.trainer import run_round  # noqa: E402

PKG_DIRS = {
    "a": "base/cpu-a-guide-20261110-v1",
    "b": "base/cpu-b-rul-20261110-v1",
    "c": "base/cpu-c-guide-20261110-v1",
}


def half_mask(y: np.ndarray, task: str, seed: int) -> np.ndarray:
    """选择集掩码：分类按类别分层各取一半，回归随机取一半。"""
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(y), dtype=bool)
    groups = [np.flatnonzero(y == c) for c in np.unique(y)] if task == "classification" \
        else [np.arange(len(y))]
    for idx in groups:
        idx = idx.copy()
        rng.shuffle(idx)
        mask[idx[: len(idx) // 2]] = True
    return mask


def make_decider(arm: str, pkg, base_lr: float, base_raw: float, epochs: int):
    if arm == "rule":
        return RulePolicy(base_epochs=epochs, base_lr=base_lr)
    llm = LLMDecider(profile=pkg.profile, task=pkg.task, metric_name=pkg.metric_name,
                     higher_better=pkg.higher_better, baseline_value=base_raw,
                     fallback=RulePolicy(base_epochs=epochs, base_lr=base_lr),
                     client=LLMClient())
    if arm == "llm":
        return llm
    return HybridDecider(AdaptivePolicy(base_epochs=epochs, base_lr=base_lr), llm)


def ratchet(pkg, Model, x, y, sel_idx, decider, work: Path, rounds: int):
    """跑 rounds 轮续训：采纳与否、LLM 看到的历史，全部只用选择集。返回 (最终预测, LLM 调用数)。"""
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    to_raw = (lambda s: s) if pkg.higher_better else (lambda s: -s)
    best_ck = None                                   # None = 随包 checkpoint（--from-baseline）
    best_pred = Model.load(pkg.checkpoint).predict(x)
    best_sel = score(best_pred, y, pkg.task, sel_idx)
    history, calls = [], 0
    for k in range(1, rounds + 1):
        if hasattr(decider, "next_parent"):
            decider.next_parent = str(best_ck)
        plan = decider.decide(history)
        if getattr(decider, "last_degraded", None) is not None:
            calls += 1
        if plan.skip:
            history.append({"round": k, "value": float("nan"), "accepted": False,
                            "lr": plan.learning_rate, "epochs": plan.epochs,
                            "parent": str(best_ck), "trained": False, "reason": "NOOP"})
            continue
        rr = run_round(pkg, work / f"round-{k}", resume=best_ck, epochs=plan.epochs,
                       stage_id=f"round-{k}", learning_rate=plan.learning_rate)
        if not rr.ok:
            history.append({"round": k, "value": float("nan"), "accepted": False,
                            "lr": plan.learning_rate, "epochs": plan.epochs,
                            "parent": str(best_ck), "trained": True, "reason": rr.error})
            continue
        pred = Model.load(rr.checkpoint).predict(x)
        s = score(pred, y, pkg.task, sel_idx)
        accepted = s > best_sel
        reason = ("通过：优于当前最优" if accepted else
                  f"候选 {to_raw(s):.4f} 未优于当前最优 {to_raw(best_sel):.4f}")
        history.append({"round": k, "value": to_raw(s), "accepted": accepted,
                        "lr": plan.learning_rate, "epochs": plan.epochs,
                        "parent": str(best_ck), "trained": True, "reason": reason})
        if accepted:
            best_ck, best_sel, best_pred = rr.checkpoint, s, pred
    return best_pred, calls


def verdict(sig) -> str:
    if sig is None:
        return "无法计算"
    if sig.ci_low <= 0.0 <= sig.ci_high:
        return "无法区分"
    return "显著更好" if sig.p_better >= 0.95 else ("显著更差" if sig.p_better <= 0.05 else "无法区分")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packages", default="b,c")
    ap.add_argument("--arms", default="rule,hybrid")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=600)
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    ref = arms[0]
    lines = ["# 决策器 split-half 裁决（选择集做决策，留出集做比较）", "",
             f"- 对照基准：`{ref}`；每个 split 各跑 {args.rounds} 轮；分数为留出集上的官方主指标。",
             "- 判定：配对 bootstrap，95% CI 跨 0 记为「无法区分」。", "",
             "| 包 | split | 臂 | 留出集主指标 | 相对基线 | 相对 " + ref + " | P(更好) | 95% CI | LLM调用 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name in [p.strip().lower() for p in args.packages.split(",") if p.strip()]:
        pkg = load_package(ROOT / PKG_DIRS[name])
        Model = _load_official_model_cls(pkg.root)
        sys.path.insert(0, str(pkg.root))
        from baselines.data import read_dataset
        cfg = json.loads(pkg.config_path().read_text())
        base_lr = float(cfg.get("learning_rate", 0.003))
        tr = read_dataset(pkg.train_jsonl, pkg.task, cfg.get("classes", []))
        va = read_dataset(pkg.validation_jsonl, pkg.task, cfg.get("classes", []), tr["paths"])
        x, y = va["x"], va["y"]
        to_raw = (lambda s: s) if pkg.higher_better else (lambda s: -s)
        base_pred = Model.load(pkg.checkpoint).predict(x)
        for seed in args.seeds:
            m = half_mask(y, pkg.task, seed)
            sel, hold = np.flatnonzero(m), np.flatnonzero(~m)
            base_raw_sel = to_raw(score(base_pred, y, pkg.task, sel))
            base_hold = score(base_pred, y, pkg.task, hold)
            print(f"==== {name} split={seed}  选择集 {len(sel)} / 留出集 {len(hold)} "
                  f" 基线留出={to_raw(base_hold):.4f}")
            preds, calls = {}, {}
            for arm in arms:
                dec = make_decider(arm, pkg, base_lr, base_raw_sel, args.epochs)
                work = ROOT / "outputs" / "adjudicate" / "deciders" / name / f"s{seed}" / arm
                preds[arm], calls[arm] = ratchet(pkg, Model, x, y, sel, dec, work, args.rounds)
            for arm in arms:
                h = score(preds[arm], y, pkg.task, hold)
                if arm == ref:
                    vs_ref, sig_txt = "—", ("—", "—", "—")
                else:
                    sig = paired_bootstrap(preds[arm][hold], preds[ref][hold], y[hold], pkg.task,
                                           bootstrap=args.bootstrap, seed=seed + 1)
                    vs_ref = verdict(sig)
                    sig_txt = ("—", "—", vs_ref) if sig is None else (
                        f"{sig.p_better:.3f}", f"[{sig.ci_low:+.4f},{sig.ci_high:+.4f}]", vs_ref)
                gain = h - base_hold                    # score 空间：越大越好
                print(f"  {arm:7s} 留出={to_raw(h):.4f}  相对基线 {gain:+.4f}  vs {ref}: {vs_ref}"
                      f"  P={sig_txt[0]} CI={sig_txt[1]}  LLM调用={calls[arm]}")
                lines.append(f"| {name} | {seed} | {arm} | {to_raw(h):.4f} | {gain:+.4f} | "
                             f"{sig_txt[2]} | {sig_txt[0]} | {sig_txt[1]} | {calls[arm]} |")
    out = ROOT / "outputs" / "ADJUDICATE_deciders.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
