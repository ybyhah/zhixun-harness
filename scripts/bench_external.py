"""外部开源基线对比评测（仅本地，不进提交镜像）。

把 river / frouros / nannyml 作为外部参照，在 A1/A2/A3 同一批次流、同一指标口径下
与现有 Harness（E0 静态基线 / E2 规则闭环）对比。本脚本不改 harness 主干，
只读取同一套适配器与指标，保证口径一致。

用法：
    python3 scripts/bench_external.py --dim online  --config configs/a1_gas.yaml
    python3 scripts/bench_external.py --dim drift    --config configs/a2_xjtu.yaml
    python3 scripts/bench_external.py --dim estimate --config configs/a1_gas.yaml
    python3 scripts/bench_external.py --dim all
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

from harness import metrics  # noqa: E402
from harness.config import load_config  # noqa: E402
from harness.datasets.factory import build_adapter  # noqa: E402
from harness.datasets.preprocess import fill_missing  # noqa: E402
from harness.monitor.performance import evaluate  # noqa: E402

# 复用 run_loop 的静态基线构造，确保对比起点一致
sys.path.insert(0, str(ROOT / "scripts"))
from run_loop import build_baseline  # noqa: E402

DEFAULT_CONFIGS = {
    "a1": ROOT / "configs" / "a1_gas.yaml",
    "a2": ROOT / "configs" / "a2_xjtu.yaml",
    "a3": ROOT / "configs" / "a3_cmapss.yaml",
}


def to_dicts(X: np.ndarray) -> list[dict]:
    """numpy 行 → river 需要的 {特征序号: 值} 字典。"""
    return [{j: float(v) for j, v in enumerate(row)} for row in X]


def baseline_per_batch(cfg, adapter, batches):
    """静态基线逐批预测，返回 [(idx, y_true, y_pred), ...]（跳过批 0）。"""
    baseline = build_baseline(cfg, adapter, batches)
    out = []
    for i, b in enumerate(batches):
        if i == 0:
            continue
        out.append((b.index, b.y, baseline.predict(fill_missing(b.X))))
    return out


def our_scores(cfg, adapter, batches):
    """用 run_loop 产出的 e2_<tag>.csv 读我们闭环的逐批 score 与 drift_type。"""
    tag = cfg.task.get("name", "task")
    p = ROOT / "outputs" / f"e2_{tag}.csv"
    if not p.exists():
        return {}, {}
    scores, drift = {}, {}
    for r in csv.DictReader(p.open(encoding="utf-8")):
        idx = int(r["batch"])
        scores[idx] = float(r["score"])
        drift[idx] = r.get("drift_type", "")
    return scores, drift


# ====================== 1.2 River 在线学习基线 ======================

def _river_models(task_type: str):
    from river import forest, tree
    if task_type == "regression":
        return {
            "HAT-reg": tree.HoeffdingAdaptiveTreeRegressor(seed=42),
            "ARF-reg": forest.ARFRegressor(n_models=5, seed=42),
        }
    return {
        "HAT-clf": tree.HoeffdingAdaptiveTreeClassifier(seed=42),
        "ARF-clf": forest.ARFClassifier(n_models=5, seed=42),
    }


def run_online(cfg, adapter, batches):
    """逐批 predict→评分→learn，严格提交后才学，不偷看未来。返回 {名称: (scores, 秒)}。"""
    metric = cfg.task.get("metric", "macro_f1")
    ttype = cfg.task.get("type", "classification")
    cls = None if ttype == "regression" else adapter.class_set
    results = {}
    for name, model in _river_models(ttype).items():
        per_batch, t0 = {}, time.perf_counter()
        for i, b in enumerate(batches):
            X = to_dicts(fill_missing(b.X))
            if i > 0:                            # 批 0 模型全空，不计入（与 E0/E2 对齐）
                preds = []
                for x in X:
                    p = model.predict_one(x)
                    if p is None:                 # 冷启动占位
                        p = 0.0 if ttype == "regression" else int(cls[0])
                    preds.append(p)
                per_batch[b.index] = evaluate(
                    b.y, np.asarray(preds), cls, metric, ttype
                ).score
            for x, y in zip(X, b.y):              # 标签在提交后才可见
                model.learn_one(x, float(y) if ttype == "regression" else int(y))
        results[name] = (per_batch, time.perf_counter() - t0)
    return results


# ====================== 1.1 漂移检测对齐 ======================

def _error_stream(batches, base_preds, task_type: str):
    """每批一个误差指标流：分类=错误率，回归=归一化 MAE。供检测器逐批喂入。"""
    out = {}
    for idx, y_true, y_pred in base_preds:
        if task_type == "regression":
            out[idx] = float(np.mean(np.abs(y_true - y_pred)))
        else:
            out[idx] = float(np.mean(y_true != y_pred))
    return out


def run_drift(cfg, adapter, batches, base_preds):
    """外部检测器 vs 我们 monitor 的 drift 判定批次。"""
    from river import drift
    from river.drift.binary import DDM, HDDMW

    ttype = cfg.task.get("type", "classification")
    _, our_drift = our_scores(cfg, adapter, batches)
    ours = {i for i, d in our_drift.items() if d not in ("STABLE", "COVARIATE_BENIGN", "")}

    # 逐样本误差流（分类 0/1，回归用是否超过训练集 MAE 的 1.5 倍）
    thr = None
    detectors = {
        "ADWIN": drift.ADWIN(), "KSWIN": drift.KSWIN(seed=42),
        "PageHinkley": drift.PageHinkley(),
    }
    bin_det = {"DDM": DDM(), "HDDM_W": HDDMW()}
    flagged = {k: set() for k in list(detectors) + list(bin_det)}
    for idx, y_true, y_pred in base_preds:
        if ttype == "regression":
            err = np.abs(y_true - y_pred)
            thr = thr if thr is not None else float(np.mean(err) * 1.5)
            stream = (err > thr).astype(float)
        else:
            stream = (y_true != y_pred).astype(float)
        for name, det in detectors.items():
            for v in stream:
                det.update(float(v))
                if det.drift_detected:
                    flagged[name].add(idx)
        for name, det in bin_det.items():
            for v in stream:
                det.update(int(v))
                if det.drift_detected:
                    flagged[name].add(idx)
    return ours, flagged


# ====================== 1.3 门控偏乐观量化 ======================

def run_estimate(cfg, adapter, batches):
    """门控分偏乐观对比：两种验证口径各自的"高估"（门控分 − 下一批真实分）。

    in_batch：候选训练用当前批 70%，门控在当前批留出 30% 上打分（现状口径）。
    holdout_recent：候选训练用上一批，门控在当前批整批上打分（时序验证口径）。
    两者的"下一批真实分"都用各自模型在下一批上的分。高估越小说明门控越可信。
    """
    from harness.models.industrial import IndustrialModel, TrainBudget
    from harness.datasets.preprocess import IncrementalScaler
    from sklearn.model_selection import train_test_split

    ttype = cfg.task.get("type", "classification")
    if ttype != "classification":
        return []                                # 本维度只做分类任务
    metric = cfg.task.get("metric", "macro_f1")
    cls = adapter.class_set

    def fit(X, y):
        X = fill_missing(X)
        scaler = IncrementalScaler(X.shape[1])
        scaler.partial_fit(X)
        m = IndustrialModel(
            class_set=cls, backend=cfg.model["backend"],
            params=dict(cfg.model.get("params", {})), seed=cfg.seed,
            task_type=ttype, budget=TrainBudget.from_config(cfg),
            scaler=scaler.copy() if cfg.data.get("standardize") else None,
        )
        m.fit(X, y)
        return m

    def score(m, X, y):
        return evaluate(y, m.predict(fill_missing(X)), cls, metric, ttype).score

    rows = []
    for i in range(1, len(batches) - 1):
        b, nxt, prev = batches[i], batches[i + 1], batches[i - 1]
        if len(np.unique(b.y)) < 2:
            continue
        # in_batch 口径
        Xtr, Xval, ytr, yval = train_test_split(
            fill_missing(b.X), b.y, test_size=0.3, random_state=42, stratify=b.y
        )
        m_in = fit(Xtr, ytr)
        gate_in = score(m_in, Xval, yval)
        next_in = score(m_in, nxt.X, nxt.y)
        # holdout_recent 口径：训练用上一批，门控打分用当前整批
        if len(np.unique(prev.y)) < 2:
            continue
        m_ho = fit(prev.X, prev.y)
        gate_ho = score(m_ho, b.X, b.y)
        next_ho = score(m_ho, nxt.X, nxt.y)
        rows.append((b.index, gate_in, next_in, gate_in - next_in,
                     gate_ho, next_ho, gate_ho - next_ho))
    return rows


# ====================== 打印与入口 ======================

def _composite(scores: dict) -> tuple[float, float]:
    vals = list(scores.values())
    return metrics.composite(vals), metrics.worst_stage(vals)


def print_online(tag, cfg, adapter, batches):
    our, _ = our_scores(cfg, adapter, batches)
    e0_comp = e0_worst = float("nan")
    print(f"\n=== [{tag}] 在线学习基线 vs E2 ===")
    res = run_online(cfg, adapter, batches)
    print(f"{'方案':<12}{'综合':>10}{'最差阶段':>10}{'耗时s':>8}")
    if our:
        c, w = _composite(our)
        print(f"{'E2 规则闭环':<12}{c:>10.4f}{w:>10.4f}{'-':>8}")
    for name, (sc, sec) in res.items():
        c, w = _composite(sc)
        print(f"{name:<12}{c:>10.4f}{w:>10.4f}{sec:>8.1f}")


def print_drift(tag, cfg, adapter, batches, base_preds):
    ours, flagged = run_drift(cfg, adapter, batches, base_preds)
    print(f"\n=== [{tag}] 漂移检测对齐（我们判定 drift 的批次：{sorted(ours)}）===")
    print(f"{'检测器':<14}{'告警批次':<28}{'命中':>5}{'漏报':>5}{'误报':>5}")
    for name, fb in flagged.items():
        hit = len(ours & fb)
        miss = len(ours - fb)
        fp = len(fb - ours)
        print(f"{name:<14}{str(sorted(fb)):<28}{hit:>5}{miss:>5}{fp:>5}")


def print_estimate(tag, cfg, adapter, batches):
    rows = run_estimate(cfg, adapter, batches)
    if not rows:
        print(f"\n=== [{tag}] 门控偏乐观量化：跳过（非分类任务或样本不足）===")
        return
    print(f"\n=== [{tag}] 门控偏乐观量化：in_batch vs holdout_recent ===")
    print(f"{'批次':<6}{'in门控':>9}{'in下批':>9}{'in高估':>9}"
          f"{'ho门控':>9}{'ho下批':>9}{'ho高估':>9}")
    for idx, gi, ni, di, gh, nh, dh in rows:
        print(f"{idx:<6}{gi:>9.3f}{ni:>9.3f}{di:>+9.3f}"
              f"{gh:>9.3f}{nh:>9.3f}{dh:>+9.3f}")
    gi = float(np.mean([r[3] for r in rows]))
    gh = float(np.mean([r[6] for r in rows]))
    print(f"平均高估：in_batch {gi:+.4f}  vs  holdout_recent {gh:+.4f}"
          f"（越接近 0 越可信）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", default="all",
                    choices=["drift", "online", "estimate", "all"])
    ap.add_argument("--config", default="")
    args = ap.parse_args()

    configs = [Path(args.config)] if args.config else list(DEFAULT_CONFIGS.values())
    for cfg_path in configs:
        cfg = load_config(str(cfg_path))
        tag = cfg.task.get("name", cfg_path.stem)
        adapter = build_adapter(cfg, ROOT)
        try:
            batches = adapter.load_batches()
        except FileNotFoundError as exc:
            print(f"\n[{tag}] 跳过：{exc}")
            continue
        base_preds = baseline_per_batch(cfg, adapter, batches)
        for dim in (["drift", "online", "estimate"] if args.dim == "all" else [args.dim]):
            try:
                if dim == "online":
                    print_online(tag, cfg, adapter, batches)
                elif dim == "drift":
                    print_drift(tag, cfg, adapter, batches, base_preds)
                else:
                    print_estimate(tag, cfg, adapter, batches)
            except ImportError as exc:
                print(f"\n[{tag}] {dim} 维度所需库未安装，跳过：{exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
