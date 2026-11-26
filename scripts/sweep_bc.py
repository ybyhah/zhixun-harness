"""B/C 续训超参扫描：从官方 checkpoint 出发，比较不同 epochs/lr/batch 在公开验证集上的 RMSE/NASA。
用法：python scripts/sweep_bc.py B
"""
import sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
from baselines.train import train_run  # noqa: E402
from baselines.model import Model  # noqa: E402
from baselines.data import read_dataset  # noqa: E402

PKG = {"B": "cpu-b-rul-20261110-v1", "C": "cpu-c-guide-20261110-v1"}


def nasa(y, p):
    d = p - y
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def main(key):
    b = ROOT / "base" / PKG[key]
    tr, va = str(b / "public/train.jsonl"), str(b / "public/validation.jsonl")
    val = read_dataset(va, "regression", [])
    m0 = Model.load(str(b / "checkpoint.npz"))
    p0 = m0.predict(val["x"])
    print("baseline", m0.meta["learning_rate"], m0.meta["batch_size"], m0.meta["epochs"],
          "rmse", float(np.sqrt(np.mean((p0 - val["y"]) ** 2))), "bias", float(np.mean(p0 - val["y"])),
          "y_mean", float(np.mean(val["y"])), "y_std", float(np.std(val["y"])), flush=True)
    grid = [(e, lr, bs) for e in (1, 3, 5) for lr in (0.0003, 0.001, 0.003, 0.01) for bs in (None,)] if key == "C" else [(e, lr, bs) for e in (1, 2, 3, 5, 8) for lr in (0.001, 0.003, 0.01, 0.02) for bs in (None,)]
    for e, lr, bs in grid:
        o = Path(tempfile.mkdtemp()) / "o"
        r = train_run(key, tr, va, o, resume=str(b / "checkpoint.npz"), epochs=e, learning_rate=lr, batch_size=bs)
        p = Model.load(str(o / "checkpoint.npz")).predict(val["x"])
        print(e, lr, bs, "rmse", round(r["validation_after"]["rmse"], 3), "mae", round(r["validation_after"]["mae"], 3),
              "bias", round(float(np.mean(p - val["y"])), 2), "nasa", f"{nasa(val['y'], p):.4g}",
              "sec", round(r["training_seconds"], 1), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
