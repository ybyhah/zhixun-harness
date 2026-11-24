#!/usr/bin/env python3
"""C 任务对照实验：完整公开验证集 vs 类隐藏集（RUL ≤ 上限）。

隐藏测试取官方测试序列的最后观测循环，真实 RUL 远低于公开验证集的 0–434。
本脚本按 harness 的 C 续训轨迹，在同一份数据、同一随机种子下对比：
训练目标是否截断 × 预测是否截断。只用官方公开包，不进提交镜像。

用法：
    python3 scripts/c_hidden_like_experiment.py \
        --package base/cpu-c-guide-20261110-v1 \
        --report-dir reports/c_hidden_alignment_2026-11-25
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
from baselines.data import read_dataset  # noqa: E402
from baselines.model import Model  # noqa: E402
from baselines.train import _nasa_score  # noqa: E402

# 与 contest_submission 的 C 轨迹一致：第 1、2 轮固定，之后单 epoch 搜索学习率。
PLAN = [(5, 0.001), (3, 0.0005), (1, 0.0004), (1, 0.0003), (1, 0.00045)]
VARIANTS = [
    ("当前做法（不截断）", None, None),
    ("预测截断 200", None, 200),
    ("训练截断 250 + 预测截断 200", 250, 200),
    ("训练截断 200 + 预测截断 200", 200, 200),
]


def evaluate(pred, y, mask):
    full = float(np.sqrt(np.mean((pred - y) ** 2)))
    sub_err = pred[mask] - y[mask]
    return {"full_rmse": full,
            "hidden_like_rmse": float(np.sqrt(np.mean(sub_err ** 2))),
            "hidden_like_mae": float(np.mean(np.abs(sub_err))),
            "hidden_like_nasa": _nasa_score(pred[mask], y[mask])}


def run(package, hidden_like_max):
    train = read_dataset(package / "public" / "train.jsonl", "regression", [])
    valid = read_dataset(package / "public" / "validation.jsonl", "regression", [])
    y = valid["y"]
    mask = y <= hidden_like_max
    baseline = Model.load(package / "checkpoint.npz")
    base_pred = baseline.predict(valid["x"])
    under = np.expm1(np.where(base_pred < y, (y - base_pred) / 13.0, 0.0)).sum()
    facts = {
        "validation_rows": int(len(y)), "hidden_like_rows": int(mask.sum()),
        "validation_rul_max": float(y.max()), "baseline_prediction_max": float(base_pred.max()),
        "baseline_nasa_underestimate_share": float(under / _nasa_score(base_pred, y)),
    }
    rows = [{"variant": "官方基线", "round": 0, **evaluate(base_pred, y, mask)}]
    for name, target_cap, prediction_cap in VARIANTS:
        model = Model.load(package / "checkpoint.npz")
        target = train["y"] if target_cap is None else np.minimum(train["y"], target_cap)
        for index, (epochs, lr) in enumerate(PLAN, 1):
            model.meta["learning_rate"] = lr
            model.fit(train["x"], target, epochs)
            pred = model.predict(valid["x"])
            if prediction_cap is not None:
                pred = np.minimum(pred, prediction_cap)
            rows.append({"variant": name, "round": index, **evaluate(pred, y, mask)})
    return facts, rows


def to_markdown(facts, rows, hidden_like_max):
    lines = ["# C 类隐藏集对照实验", "",
             f"类隐藏集：公开验证集中真实 RUL ≤ {hidden_like_max} 的样本。", "",
             "## 数据事实", ""]
    lines += [f"- {key}: {value:.4g}" if isinstance(value, float) else f"- {key}: {value}"
              for key, value in facts.items()]
    lines += ["", "## 逐轮结果", "",
              "| 方案 | 轮 | 完整公开集 RMSE | 类隐藏集 RMSE | 类隐藏集 MAE | 类隐藏集 NASA |",
              "|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['variant']} | {row['round']} | {row['full_rmse']:.2f} | "
                     f"{row['hidden_like_rmse']:.2f} | {row['hidden_like_mae']:.2f} | "
                     f"{row['hidden_like_nasa']:.3e} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path,
                        default=ROOT / "base" / "cpu-c-guide-20261110-v1")
    parser.add_argument("--report-dir", type=Path,
                        default=ROOT / "reports" / "c_hidden_alignment_2026-11-25")
    parser.add_argument("--hidden-like-max", type=float, default=200.0)
    args = parser.parse_args()
    facts, rows = run(args.package, args.hidden_like_max)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "results.json").write_text(
        json.dumps({"facts": facts, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    markdown = to_markdown(facts, rows, args.hidden_like_max)
    (args.report_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
