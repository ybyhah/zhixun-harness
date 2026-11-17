"""E2 规则闭环实验，与 E0 静态基线对照，并支持消融。

用法：
    python scripts/run_loop.py                        # 默认配置
    python scripts/run_loop.py --variant no-compete    # 关掉多候选竞争
    python scripts/run_loop.py --set gate.forget_tol=1.0
    python scripts/run_loop.py --sweep                 # 跑全部消融组合

产出：
    outputs/e2_rule_loop.csv    逐批次指标与动作
    outputs/audit.jsonl         审计记录（含闭环证据）
    outputs/curve_tuning.png    E0 / 优化前 E2 / 当前 E2 对照曲线
    outputs/summary_tuning.csv  各组合的统一口径指标
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harness import metrics  # noqa: E402
from harness.config import Config, load_config  # noqa: E402
from harness.datasets.factory import build_adapter  # noqa: E402
from harness.datasets.preprocess import IncrementalScaler, fill_missing  # noqa: E402
from harness.decide.rules import RuleDecider  # noqa: E402
from harness.loop import SelfEvolvingLoop  # noqa: E402
from harness.models.industrial import IndustrialModel, TrainBudget  # noqa: E402

# 消融变体：各关掉一项优化，用于拆解单项贡献
VARIANTS: dict[str, dict] = {
    "full": {},
    "no-budget": {"budget.enabled": False},
    "no-window-min": {"loop.min_window_samples": 0},
    "no-compete": {"loop.max_candidates": 1},
    "no-forget-check": {"gate.forget_tol": 1.0},
    "no-min-gain": {"gate.min_gain": 0.0},
    "holdout-gate": {"gate.val_mode": "holdout_recent"},
    "holdout-refit": {"gate.val_mode": "holdout_refit"},
}

def override(cfg: Config, dotted: dict) -> Config:
    """按点号路径覆盖配置，用于消融。"""
    raw = copy.deepcopy(cfg.raw)
    for path, value in dotted.items():
        node = raw
        parts = path.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return Config(raw=raw)


def parse_set(items: list[str]) -> dict:
    out: dict = {}
    for item in items or []:
        key, _, val = item.partition("=")
        if val.lower() in ("true", "false"):
            parsed: object = val.lower() == "true"
        else:
            try:
                parsed = int(val)
            except ValueError:
                try:
                    parsed = float(val)
                except ValueError:
                    parsed = val
        out[key.strip()] = parsed
    return out


def build_baseline(cfg, adapter, batches) -> IndustrialModel:
    """静态基线：只用首批数据训练，之后不更新。门控第二道检查会用到它。"""
    scaler = IncrementalScaler(batches[0].X.shape[1])
    X0 = fill_missing(batches[0].X)
    scaler.partial_fit(X0)
    model = IndustrialModel(
        class_set=adapter.class_set,
        backend=cfg.model["backend"],
        params=dict(cfg.model.get("params", {})),
        seed=cfg.seed,
        task_type=cfg.task.get("type", "classification"),
        budget=TrainBudget.from_config(cfg),
        scaler=scaler.copy() if cfg.data.get("standardize") else None,
    )
    model.fit(X0, batches[0].y)
    return model


def run_one(cfg, adapter, batches, baseline, label: str):
    decider = RuleDecider(
        min_batch_size=cfg.loop.get("min_batch_size", 30),
        default_window=3,
    )
    loop = SelfEvolvingLoop(
        batches, adapter.class_set, cfg, decider, baseline_model=baseline
    )
    res = loop.run()
    summary = metrics.summarize(
        res.scores,
        trainings=res.total_trainings,
        seconds=res.wall_seconds,
        replay_pairs=res.replay_pairs,
        tol=float(cfg.monitor.get("perf_drop", 0.05)),
        worst_window=int(cfg.get("metrics.worst_window", 2)),
        promotions=res.promotions,
        offline_batches=len(res.offline_batches),
        rollbacks=len(res.rollbacks),
    )
    return res, summary, label

def read_scores(path: str | Path) -> dict[int, float]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as fh:
        out = {}
        for r in csv.DictReader(fh):
            try:
                out[int(r["batch"])] = float(r["score"])
            except (ValueError, KeyError):
                continue
        return out


def compute_e0(baseline, batches, cfg, critical: dict | None = None):
    """用冻结的静态基线逐批预测，返回 {batch_index: 统一score}。

    传入 critical 字典时，顺带填入每批关键类召回（task.critical_class 未配置则不填）。
    """
    from harness.datasets.preprocess import fill_missing
    from harness.monitor.performance import evaluate
    metric = cfg.task.get("metric", "macro_f1")
    ttype = cfg.task.get("type", "classification")
    crit_cls = cfg.task.get("critical_class")
    cls = None if ttype == "regression" else baseline.class_set
    out = {}
    for i, b in enumerate(batches):
        if i == 0:
            continue  # 基线用首批训练，不计入对照
        perf = evaluate(b.y, baseline.predict(fill_missing(b.X)), cls, metric, ttype,
                        crit_cls)
        out[b.index] = perf.score
        if critical is not None and crit_cls is not None:
            critical[b.index] = perf.critical_recall
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "a1_gas.yaml"))
    ap.add_argument("--variant", default="full", choices=sorted(VARIANTS))
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--sweep", action="store_true", help="跑全部消融组合")
    ap.add_argument("--swanlab", action="store_true",
                    help="启用 SwanLab 记录（等价 record.swanlab.enabled=true）")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    base_cfg = load_config(args.config)
    if args.swanlab:
        base_cfg = override(base_cfg, {"record.swanlab.enabled": True})
    adapter = build_adapter(base_cfg, ROOT)
    batches = adapter.load_batches()
    metric = base_cfg.task.get("metric", "macro_f1")

    print("训练静态基线 ...")
    baseline = build_baseline(base_cfg, adapter, batches)
    e0_crit: dict[int, float] = {}
    e0 = compute_e0(baseline, batches, base_cfg, e0_crit)
    e0_composite = metrics.composite(list(e0.values())) if e0 else float("nan")
    tag = base_cfg.task.get("name", "task")

    plan = sorted(VARIANTS) if args.sweep else [args.variant]
    results = []
    for name in plan:
        cfg = override(base_cfg, VARIANTS[name])
        cfg = override(cfg, parse_set(args.sets))
        cfg = override(cfg, {"record.swanlab.run_name": f"{tag}-{name}"})
        if args.sweep:
            # 消融时各自写独立的审计文件，避免互相覆盖
            cfg = override(cfg, {"record.audit_path": f"outputs/audit_{name}.jsonl"})
        print(f"\n运行变体 {name} ...")
        res, summary, label = run_one(cfg, adapter, batches, baseline, name)
        summary.gain_vs_baseline = (
            summary.composite - e0_composite
            if e0_composite == e0_composite else float("nan")
        )
        t, s = metrics.resource_efficiency(
            summary.gain_vs_baseline, summary.trainings, summary.seconds
        )
        summary.trainings_per_gain, summary.seconds_per_gain = t, s
        results.append((label, res, summary))

    # ---- 主变体的逐批输出 ----
    main_label, main_res, main_sum = results[0]
    out = ROOT / "outputs" / f"e2_{tag}.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(main_res.rows[0].keys()))
        w.writeheader()
        w.writerows(main_res.rows)

    shared = sorted(set(e0) & {r["batch"] for r in main_res.rows})
    if shared:
        crit_hdr = f"{'E0关键召回':>12}{'E2关键召回':>12}" if e0_crit else ""
        print(f"\n{'批次':<6}{'E0':>9}{'E2':>9}{'差值':>9}{crit_hdr}  动作 / 漂移判定")
        for i in shared:
            row = next(r for r in main_res.rows if r["batch"] == i)
            crit = (f"{e0_crit.get(i, float('nan')):>12.3f}"
                    f"{row['critical_recall']:>12.3f}") if e0_crit else ""
            print(f"{i:<6}{e0[i]:>9.4f}{row['score']:>9.4f}"
                  f"{row['score'] - e0[i]:>+9.4f}{crit}  "
                  f"{row['action']} / {row['drift_type']}")
        if e0_crit:
            e2c = [r["critical_recall"] for r in main_res.rows if r["batch"] in shared]
            print(f"关键类平均召回（跳过未出现的批次）：E0 "
                  f"{np.nanmean(list(e0_crit.values())):.3f} / E2 {np.nanmean(e2c):.3f}")

    # ---- 指标表 ----
    header = (f"\n{'变体':<16}{'综合性能':>10}{'最差阶段':>10}{'恢复':>8}"
              f"{'抗遗忘':>9}{'训练':>6}{'上线':>6}{'耗时s':>8}")
    print(header)
    print(f"{'E0 静态基线':<16}{e0_composite:>10.4f}"
          f"{metrics.worst_stage(list(e0.values())):>10.4f}"
          f"{'未恢复':>8}{'-':>9}{1:>6}{0:>6}{'-':>8}")
    rows_out = []
    for label, res, sm in results:
        print(f"{label:<16}{sm.composite:>10.4f}{sm.worst_stage:>10.4f}"
              f"{sm.recovery_text():>8}"
              f"{(f'{sm.forgetting:+.3f}' if sm.forgetting == sm.forgetting else '-'):>9}"
              f"{sm.trainings:>6}{sm.promotions:>6}{sm.seconds:>8.1f}")
        rows_out.append({"变体": label, **sm.to_dict()})

    with (ROOT / "outputs" / f"summary_{tag}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    print(f"\n完整闭环记录 {len(main_res.audit.complete_loops())} 条（门槛 >= 3）")
    print(f"护栏拦截 {len(main_res.guard_violations)}，降级事件 "
          f"{len(main_res.failures)}，单批最大 "
          f"{max(r['seconds'] for r in main_res.rows):.1f}s")

    if not args.no_plot and shared:
        plot(shared, e0, main_res, ROOT / "outputs" / f"curve_{tag}.png", metric)
    return 0


def plot(idx, e0, res, path: Path, metric: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    before = read_scores(ROOT / "outputs" / "e2_before_tuning.csv") if "a1" in path.stem else {}
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(idx, [e0[i] for i in idx], "o--", color="#999",
            label="E0 static baseline")
    if before:
        ax.plot(idx, [before.get(i, np.nan) for i in idx], "s:", color="#ff7f0e",
                label="E2 before tuning (in-sample gate)")
    cur = {r["batch"]: r["score"] for r in res.rows}
    ax.plot(idx, [cur.get(i, np.nan) for i in idx], "o-", color="#1f77b4",
            label="E2 after tuning (held-out gate)")
    for r in res.rows:
        if r["batch"] in idx and r["gate_accepted"]:
            ax.axvline(r["batch"], color="#2ca02c", alpha=0.22, lw=6)
    ax.plot([], [], color="#2ca02c", alpha=0.3, lw=6, label="model updated")
    ax.set_xlabel("batch")
    ax.set_ylabel(f"unified score (higher=better; reg = -{metric})")
    ax.set_title(f"{path.stem}: static baseline vs rule-based loop")
    ax.set_xticks(idx)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"曲线已保存 {path}")


if __name__ == "__main__":
    raise SystemExit(main())


