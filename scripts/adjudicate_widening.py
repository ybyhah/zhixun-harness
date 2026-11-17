#!/usr/bin/env python3
"""用 split-half 协议裁决"加宽是否真有提升"（A 包）。

背景：先前用肉眼估噪声把 A 的加宽判成"无效"，随后配对 bootstrap 又显示 hidden-128
相对交付物 P(更好)=0.988。两个判断都不可靠——前者欠检验，后者没校正选择偏差
（搜了多个宽度/超参才挑出赢家）。

正确做法：把公开验证集按类别分层对半，
  - 前一半（选择集）：各宽度各自跑 10 轮续训，**只用它**做"是否采纳本轮"的决策
  - 后一半（留出集）：完全不参与任何选择，只在最后用来比较各宽度的产出
这样比较的是"procedure(宽度)"而不是"我挑出来的那个 checkpoint"，不受选择偏差污染。

用法：
    python3 scripts/adjudicate_widening.py                 # 宽度 32/64/128
    python3 scripts/adjudicate_widening.py --widths 32 128
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from driver.migrate import _load_official_model_cls, widen_checkpoint  # noqa: E402
from driver.official_metric import score  # noqa: E402
from driver.package import load_package  # noqa: E402
from driver.significance import paired_bootstrap  # noqa: E402

PKG_DIR = "base/cpu-a-guide-20261110-v1"


def stratified_half(y: np.ndarray, seed: int = 0) -> np.ndarray:
    """按类别分层取一半作为"选择集"，返回布尔掩码。"""
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(y), dtype=bool)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        mask[idx[: len(idx) // 2]] = True
    return mask


def ratchet(pkg, Model, va, sel_idx, width, work: Path, *, rounds=10, epochs=5,
            base_lr=0.003, min_lr=1e-5) -> np.ndarray:
    """某宽度下跑 rounds 轮续训，只用 sel_idx 上的指标决定是否采纳。返回最终预测。"""
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    cfg = json.loads(pkg.config_path().read_text())

    if width == cfg["hidden"]:
        start, profile = pkg.checkpoint, pkg.profile          # 原始宽度
    else:
        start = work / "migrated.npz"
        widen_checkpoint(pkg.root, pkg.checkpoint, start, width)
        cfg["hidden"] = width
        profile = work / f"A-h{width}.json"
        profile.write_text(json.dumps(cfg, ensure_ascii=False))

    best_ck = Path(start)
    best_pred = Model.load(best_ck).predict(va["x"])
    best_sel = score(best_pred, va["y"], pkg.task, sel_idx)
    lr = base_lr
    for k in range(1, rounds + 1):
        out = work / f"round-{k}"
        cmd = [sys.executable, "-m", "baselines.train", "--profile", str(profile),
               "--train", str(pkg.train_jsonl), "--validation", str(pkg.validation_jsonl),
               "--resume", str(best_ck.resolve()), "--epochs", str(epochs),
               "--learning-rate", str(lr), "--output", str(out.resolve()),
               "--stage-id", f"round-{k}"]
        proc = subprocess.run(cmd, cwd=str(pkg.root), capture_output=True, text=True)
        if proc.returncode != 0:
            continue
        pred = Model.load(out / "checkpoint.npz").predict(va["x"])
        sel = score(pred, va["y"], pkg.task, sel_idx)
        if sel > best_sel:                       # 只看选择集
            best_ck, best_sel, best_pred = out / "checkpoint.npz", sel, pred
        else:
            lr = max(lr * 0.5, min_lr)           # 与驱动器的规则策略一致
    return best_pred


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--bootstrap", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pkg = load_package(ROOT / PKG_DIR)
    Model = _load_official_model_cls(pkg.root)
    sys.path.insert(0, str(pkg.root))
    from baselines.data import read_dataset

    cfg = json.loads(pkg.config_path().read_text())
    tr = read_dataset(pkg.train_jsonl, pkg.task, cfg["classes"])
    va = read_dataset(pkg.validation_jsonl, pkg.task, cfg["classes"], tr["paths"])
    y = va["y"]
    sel_mask = stratified_half(y, args.seed)
    sel_idx, hold_idx = np.flatnonzero(sel_mask), np.flatnonzero(~sel_mask)
    print(f"验证集 {len(y)} 行 → 选择集 {len(sel_idx)} / 干净留出集 {len(hold_idx)}")
    print(f"每个宽度各跑 {args.rounds} 轮，采纳决策只看选择集\n")

    work_root = ROOT / "outputs" / "adjudicate" / "widen-a"
    shutil.rmtree(work_root, ignore_errors=True)
    results = {}
    for w in args.widths:
        pred = ratchet(pkg, Model, va, sel_idx, w, work_root / f"h{w}",
                       rounds=args.rounds)
        results[w] = pred
        print(f"  hidden={w:<4} 选择集={score(pred,y,pkg.task,sel_idx):.4f}  "
              f"干净留出集={score(pred,y,pkg.task,hold_idx):.4f}")

    ref_w = args.widths[0]
    print(f"\n留出集上的配对检验（基准 = hidden={ref_w}，即不加宽）")
    for w in args.widths[1:]:
        sig = paired_bootstrap(results[w][hold_idx], results[ref_w][hold_idx],
                               y[hold_idx], pkg.task,
                               bootstrap=args.bootstrap, seed=args.seed + 1)
        # 判定顺序要紧：先看置信区间是否跨过 0。两个模型预测完全相同时
        # p_better=0（从不"严格更好"），若只看 p 会误判成"显著更差"。
        if sig.ci_low <= 0.0 <= sig.ci_high:
            verdict = "统计上无法区分"
        elif sig.p_better >= 0.95:
            verdict = "显著更好"
        elif sig.p_better <= 0.05:
            verdict = "显著更差"
        else:
            verdict = "统计上无法区分"
        print(f"  hidden={w:<4} P(更好)={sig.p_better:.3f}  "
              f"CI=[{sig.ci_low:+.4f},{sig.ci_high:+.4f}]  均值差={sig.mean_diff:+.4f}"
              f"  -> {verdict}")
    print(f"\n中间产物：{work_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
