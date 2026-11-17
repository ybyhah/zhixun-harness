"""E0 静态基线：批次 1 训练，之后只预测不更新。

这是赛题的排名门槛参照（"综合性能不低于官方静态基线"），
也是后续所有对照实验的基准线。
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harness.config import load_config  # noqa: E402
from harness.datasets.gas_sensor import GasSensorAdapter  # noqa: E402
from harness.datasets.preprocess import IncrementalScaler, fill_missing  # noqa: E402
from harness.models.industrial import IndustrialModel, TrainBudget  # noqa: E402
from harness.monitor.performance import evaluate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "a1_gas.yaml"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "e0_baseline.csv"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    adapter = GasSensorAdapter(ROOT / cfg.data["raw_dir"], cfg.data.get("n_batches", 10))
    batches = adapter.load_batches()
    metric = cfg.task.get("metric", "macro_f1")

    # 只用第一个批次训练，scaler 也只见第一个批次
    scaler = IncrementalScaler(batches[0].X.shape[1])
    X0 = fill_missing(batches[0].X)
    scaler.partial_fit(X0)
    model = IndustrialModel(
        class_set=adapter.class_set,
        backend=cfg.model["backend"],
        params=dict(cfg.model.get("params", {})),
        seed=cfg.seed,
        budget=TrainBudget.from_config(cfg),
        scaler=scaler.copy() if cfg.data.get("standardize") else None,
    )
    t0 = time.perf_counter()
    stats = model.fit(X0, batches[0].y)
    train_seconds = time.perf_counter() - t0
    print(f"基线训练：{stats.n_samples} 样本，{train_seconds:.2f}s")

    rows = []
    for b in batches[1:]:
        t = time.perf_counter()
        X = fill_missing(b.X)
        perf = evaluate(b.y, model.predict(X), adapter.class_set, metric)
        rows.append({
            "batch": b.index,
            "n_samples": perf.n_samples,
            "score": perf.value,
            "accuracy": perf.accuracy,
            "macro_f1": perf.macro_f1,
            "classes_missing": " ".join(map(str, perf.classes_missing)),
            "seconds": time.perf_counter() - t,
        })
        print(
            f"  批次 {b.index}: {metric}={perf.value:.4f} "
            f"acc={perf.accuracy:.4f} n={perf.n_samples}"
            + (f" 缺类={perf.classes_missing}" if perf.classes_missing else "")
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    scores = np.array([r["score"] for r in rows])
    print(f"\n写入 {out}")
    print(f"综合（均值）: {scores.mean():.4f}")
    print(f"最差批次    : {scores.min():.4f} (批次 {rows[int(scores.argmin())]['batch']})")
    print(f"首批之后下降 : {scores[0] - scores[-1]:+.4f}")
    print(f"单批预测耗时 : 最大 {max(r['seconds'] for r in rows):.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
