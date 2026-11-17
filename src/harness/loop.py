"""主闭环编排。

单批次七步（doc.md 第 2.2 节）：
  1. 预测并提交  2. 取得标签  3. 监测  4. 决策
  5. 训练候选    6. 门控上线/回滚  7. 记录

第 1 步永远先执行且不被后续任何异常影响——"无人干预跑完全部阶段"是排名门槛。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .datasets.base import Batch
from .datasets.preprocess import IncrementalScaler, fill_missing
from .decide.actions import ACTION_COST, Action, ActionExecutor, Decision, TrainingData
from .gate.evaluator import Gate
from .gate.replay import ReplayBuffer
from .models.industrial import IndustrialModel, TrainBudget
from .models.registry import ModelRegistry
from .monitor.distribution import DistributionMonitor, label_shift, target_shift
from .monitor.error_stream import ErrorStreamMonitor
from .monitor.performance import evaluate
from .monitor.report import classify
from .record.audit import AuditLog
from .record.schema import DecisionRecord
from .record.swan import SwanRecorder
from .stream.batch_stream import OFFLINE, BatchStream


@dataclass
class Guard:
    """安全护栏：防止无限制迭代，对应"受控"要求。"""

    max_trainings_per_batch: int = 3
    max_total_trainings: int = 60
    train_timeout_s: float = 60.0
    total_trainings: int = 0
    violations: list[dict] = field(default_factory=list)

    def can_train(self, batch_index: int, used_in_batch: int) -> tuple[bool, str]:
        if used_in_batch >= self.max_trainings_per_batch:
            return False, f"本批训练次数已达上限 {self.max_trainings_per_batch}"
        if self.total_trainings >= self.max_total_trainings:
            return False, f"累计训练次数已达上限 {self.max_total_trainings}"
        return True, ""

    def note(self, batch_index: int, reason: str) -> None:
        self.violations.append({"batch_index": batch_index, "reason": reason})


@dataclass
class LoopResult:
    rows: list[dict[str, Any]] = field(default_factory=list)
    audit: AuditLog | None = None
    offline_batches: list[int] = field(default_factory=list)
    rollbacks: list[dict] = field(default_factory=list)
    guard_violations: list[dict] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)
    replay_pairs: list[tuple[float, float]] = field(default_factory=list)
    total_trainings: int = 0
    wall_seconds: float = 0.0

    @property
    def scores(self) -> list[float]:
        return [r["score"] for r in self.rows]

    @property
    def promotions(self) -> int:
        return sum(1 for r in self.rows[1:] if r["gate_accepted"])

    @property
    def mean_score(self) -> float:
        vals = [r["score"] for r in self.rows if r["score"] == r["score"]]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def worst_score(self) -> float:
        vals = [r["score"] for r in self.rows if r["score"] == r["score"]]
        return float(np.min(vals)) if vals else float("nan")


class SelfEvolvingLoop:
    def __init__(
        self,
        batches: list[Batch],
        class_set: np.ndarray,
        cfg,
        decider,
        baseline_model: IndustrialModel | None = None,
        audit_reset: bool = True,
    ) -> None:
        self.batches = batches
        self.class_set = class_set
        self.cfg = cfg
        self.decider = decider
        self.baseline = baseline_model

        self.metric = cfg.task.get("metric", "macro_f1")
        self.task_type = cfg.task.get("type", "classification")
        self.critical_class = cfg.task.get("critical_class")
        self.stream = BatchStream(batches)
        self.registry = ModelRegistry()
        self.pool = TrainingData()
        self.scaler = IncrementalScaler(batches[0].X.shape[1])
        self.budget = TrainBudget.from_config(cfg)
        self.executor = ActionExecutor(
            class_set,
            cfg.model["backend"],
            cfg.model.get("params", {}),
            cfg.seed,
            budget=self.budget,
            min_window_samples=cfg.loop.get("min_window_samples", 0),
            scaler_factory=self._scaler_snapshot,
            task_type=self.task_type,
        )
        self.dist_monitor = DistributionMonitor(**_dist_kwargs(cfg))
        self.err_monitor = ErrorStreamMonitor(cfg.monitor.get("error_detector", "adwin"))
        self.replay = ReplayBuffer(
            per_batch=cfg.gate.get("replay_per_batch", 60),
            capacity=cfg.gate.get("replay_capacity", 600),
            seed=cfg.seed,
        )
        self.gate = Gate(
            class_set,
            metric=self.metric,
            min_gain=cfg.gate.get("min_gain", 0.005),
            forget_tol=cfg.gate.get("forget_tol", 0.02),
            task_type=self.task_type,
            critical_class=self.critical_class,
            critical_recall_tol=cfg.gate.get("critical_recall_tol"),
        )
        self.guard = Guard(
            max_trainings_per_batch=cfg.loop.get("max_trainings_per_batch", 3),
            max_total_trainings=cfg.loop.get("max_total_trainings", 60),
            train_timeout_s=float(cfg.loop.get("train_timeout_s", 60)),
        )
        self.audit = AuditLog(cfg.record["audit_path"], reset=audit_reset)
        self.swan = SwanRecorder(
            enabled=bool(cfg.get("record.swanlab.enabled", False)),
            project=cfg.get("record.swanlab.project", "zhixun-harness"),
            mode=cfg.get("record.swanlab.mode", "offline"),
            public=bool(cfg.get("record.swanlab.public", True)),
            run_name=cfg.get("record.swanlab.run_name", None) or cfg.task.get("name"),
            config={
                "task": cfg.task.get("name"),
                "metric": self.metric,
                "task_type": self.task_type,
                "seed": cfg.seed,
                "model_backend": cfg.model.get("backend"),
                "decay": cfg.model.get("decay"),
                "val_mode": cfg.gate.get("val_mode", "in_batch"),
            },
        )
        self._recent_scores: list[float] = []
        self._version_scores: dict[int, list[float]] = {}
        self._result = LoopResult(audit=self.audit)

    # ------------------------------------------------------------------
    def run(self) -> LoopResult:
        t0 = time.perf_counter()
        while (view := self.stream.next_batch()) is not None:
            self._process(view)
        self._result.offline_batches = list(self.stream.offline_batches)
        self._result.rollbacks = list(self.registry.rollbacks)
        self._result.guard_violations = list(self.guard.violations)
        self._result.total_trainings = self.guard.total_trainings
        self._result.wall_seconds = time.perf_counter() - t0
        self.swan.finish()
        return self._result

    # ------------------------------------------------------------------
    def _process(self, view) -> None:
        idx = view.index
        t_batch = time.perf_counter()
        X_raw = fill_missing(view.X)

        # ---- 1. 预测并提交：最优先，且与后续逻辑隔离 ----
        offline = False
        pred = None
        fallback = (
            np.zeros(X_raw.shape[0], dtype=np.float64)
            if self.task_type == "regression"
            else np.full(X_raw.shape[0], int(self.class_set[0]))
        )
        try:
            if self.registry.has_online:
                pred = self.registry.online.predict(X_raw)
            else:
                pred = fallback.copy()          # 冷启动占位
        except Exception as exc:  # noqa: BLE001
            self._fail(idx, "predict", exc)
            pred = fallback.copy()

        submit = OFFLINE if offline else pred
        res = self.stream.submit(submit)
        y_true = res.y_true

        # ---- 2 到 7 步：任何异常都只降级，不中断闭环 ----
        try:
            self._after_submit(idx, X_raw, y_true, pred, t_batch)
        except Exception as exc:  # noqa: BLE001
            self._fail(idx, "post_submit", exc)
            self._result.rows.append(
                self._row(idx, y_true, pred, Decision(Action.NOOP, reason="异常降级"),
                          {}, 0.0, time.perf_counter() - t_batch)
            )

    def _after_submit(self, idx, X_raw, y_true, pred, t_batch) -> None:
        # ---- 3. 监测 ----
        # 监测与数据池都用原始特征：PSI/KS 基于分位数分箱，与量纲无关；
        # 标准化参数属于模型自身，不应渗入监测与存储。
        perf = evaluate(y_true, pred, self.class_set, self.metric, self.task_type,
                        self.critical_class)
        dist = self.dist_monitor.compare(X_raw)
        err = self.err_monitor.update(y_true, pred)
        # 参照系是"模型见过的全部数据"，而不是上一个批次。
        # A1 相邻批次构成本身波动很大，用上一批会把正常波动误判成漂移。
        ref_y = (
            np.concatenate(self.pool.y) if self.pool.n_batches else y_true
        )
        if self.task_type == "regression":
            ls = target_shift(ref_y, y_true)
        else:
            ls = label_shift(ref_y, y_true, self.class_set)
        baseline_perf = self._reference_score(perf.score)
        report = classify(
            idx, perf, dist, err, baseline_perf, ls,
            perf_drop=self.cfg.monitor.get("perf_drop", 0.05),
            label_shift_threshold=self.cfg.monitor.get("label_shift_threshold", 0.2),
            task_type=self.task_type,
        )

        # 标准化统计量随之更新，始终只用已见数据
        self.scaler.partial_fit(X_raw)
        X_now = X_raw
        self.pool.add(X_now, y_true, idx)

        serving_version = (
            self.registry.online_version.version
            if self.registry.online_version else 0
        )

        # ---- 冷启动 ----
        if not self.registry.has_online:
            self._bootstrap(idx, X_now, y_true, perf, report, t_batch)
            return

        # ---- 4. 决策（读历史）----
        history = self.audit.records
        history_read = [r.batch_index for r in history[-5:]]
        decision = self.decider.decide(report, history)

        # ---- 5 到 6. 训练候选并门控 ----
        experiments: list[dict] = []
        gate_dict: dict = {}
        used = 0
        train_pool, val_X, val_y = self.pool.split_last(
            val_frac=float(self.cfg.gate.get("val_frac", 0.3)), seed=self.cfg.seed,
            val_mode=self.cfg.gate.get("val_mode", "in_batch"),
        )
        # holdout_refit：门控用 probe（train_pool）诚实打分，上线前在全窗口重训
        refit_pool = (
            self.pool
            if self.cfg.gate.get("val_mode") == "holdout_refit" and self.pool.n_batches >= 2
            else None
        )
        if decision.action not in (Action.NOOP, Action.OFFLINE):
            limit = int(self.cfg.loop.get("max_candidates", 1))
            plan = (
                self.decider.decide_many(report, history, limit)
                if hasattr(self.decider, "decide_many")
                else [decision]
            )
            decision, gate_dict, experiments, used = self._compete(
                idx, plan, train_pool, val_X, val_y, decision, refit_pool
            )

        # ---- 回放集与参考窗口维护 ----
        self.replay.add(X_now, y_true, idx)
        if not report.perf_dropped or gate_dict.get("accepted"):
            self.dist_monitor.set_reference(X_now)

        self._note_score(perf.score, serving_version)
        self._record(idx, report, decision, history_read, experiments, gate_dict,
                     perf, time.perf_counter() - t_batch)
        self._result.rows.append(
            self._row(idx, y_true, pred, decision, gate_dict,
                      ACTION_COST[decision.action], time.perf_counter() - t_batch,
                      report)
        )

    # ------------------------------------------------------------------
    def _bootstrap(self, idx, X, y, perf, report, t_batch) -> None:
        """首批：训练初始在线模型，不经过门控。"""
        try:
            model = IndustrialModel(
                class_set=self.class_set,
                backend=self.cfg.model["backend"],
                params=dict(self.cfg.model.get("params", {})),
                seed=self.cfg.seed,
                task_type=self.task_type,
                budget=self.budget,
                scaler=self._scaler_snapshot(),
            )
            model.fit(X, y)
            self.guard.total_trainings += 1
            self.registry.bootstrap(model, idx)
            self.dist_monitor.set_reference(X)
            self.err_monitor.reset()
            self.swan.log(
                {"train_samples": model.stats.n_samples,
                 "train_seconds": model.stats.seconds}, step=idx
            )
        except Exception as exc:  # noqa: BLE001
            self._fail(idx, "bootstrap", exc)

        self.replay.add(X, y, idx)
        self._note_score(perf.score, 0)
        decision = Decision(Action.FULL_RETRAIN, reason="冷启动：用首批数据训练初始模型")
        self._record(
            idx, report, decision, [],
            [{"action": "BOOTSTRAP", "note": "首批无在线模型，直接上线"}],
            {"accepted": True, "reason": "冷启动不经门控"},
            perf, time.perf_counter() - t_batch,
        )
        self._result.rows.append(
            self._row(idx, y, None, decision, {"accepted": True}, 0,
                      time.perf_counter() - t_batch, report)
        )

    def _try_action(self, idx, decision, pool=None) -> tuple[IndustrialModel | None, dict]:
        """执行动作产出候选。训练超时或异常都只降级。"""
        exp: dict[str, Any] = {
            "action": decision.action.value,
            "params": decision.params,
        }
        t0 = time.perf_counter()
        try:
            cand = self.executor.execute(
                decision, pool if pool is not None else self.pool,
                self.registry.online, idx,
                float(self.cfg.model.get("decay", 0.9)),
            )
        except Exception as exc:  # noqa: BLE001
            self._fail(idx, f"train:{decision.action.value}", exc)
            exp["error"] = f"{type(exc).__name__}: {exc}"
            return None, exp

        elapsed = time.perf_counter() - t0
        exp["seconds"] = elapsed
        self.guard.total_trainings += 1
        if elapsed > self.guard.train_timeout_s:
            self.guard.note(
                idx,
                f"训练耗时 {elapsed:.1f}s 超过上限 "
                f"{self.guard.train_timeout_s:.0f}s，丢弃候选",
            )
            exp["timeout"] = True
            return None, exp
        if cand is not None:
            self.swan.log(
                {"train_samples": cand.stats.n_samples,
                 "train_seconds": cand.stats.seconds}, step=idx
            )
        return cand, exp

    def _compete(self, idx, plan, train_pool, val_X, val_y, fallback, refit_pool=None):
        """在训练预算内依次训练多个候选，门控择优上线。

        refit_pool 非空时（holdout_refit）：门控用 probe（在 train_pool 上训、在
        时序留出的 val 上诚实打分）判定；选中动作后在 refit_pool（全窗口）重训出
        部署模型再上线，兼顾诚实门控与最新批适应。重训失败则回退 probe 候选。

        返回 (最终采用的决策, 门控结果, 实验记录, 已用训练次数)。
        """
        experiments: list[dict] = []
        used = 0
        scored: list[tuple[float, Decision, Any, dict]] = []

        for decision in plan:
            ok, why = self.guard.can_train(idx, used)
            if not ok:
                self.guard.note(idx, why)
                if not experiments:
                    return (
                        Decision(Action.NOOP, reason=f"护栏拦截：{why}"),
                        {}, experiments, used,
                    )
                break

            cand, exp = self._try_action(idx, decision, train_pool)
            used += 1
            if cand is None:
                experiments.append(exp)
                continue

            result = self._judge(cand, val_X, val_y)
            exp.update({
                "candidate_score": result.candidate_score,
                "online_score": result.online_score,
                "gate_accepted": result.accepted,
                "gate_reason": result.reason,
            })
            experiments.append(exp)
            scored.append((result.candidate_score, decision, cand, result.to_dict()))

        passed = [t for t in scored if t[3].get("accepted")]
        if passed:
            # 多个候选都通过时取门控验证分最高者（门控基于 probe，诚实不乐观）
            _, decision, cand, gate_dict = max(
                passed, key=lambda t: (t[0] if t[0] == t[0] else -1.0)
            )
            # holdout_refit：对选中动作在全窗口重训出部署模型；失败回退 probe
            if refit_pool is not None:
                ok, why = self.guard.can_train(idx, used)
                if ok:
                    deploy, refit_exp = self._try_action(idx, decision, refit_pool)
                    used += 1
                    refit_exp["note"] = "holdout_refit 全窗口重训部署模型"
                    experiments.append(refit_exp)
                    if deploy is not None:
                        cand = deploy
                else:
                    self.guard.note(idx, f"重训受护栏拦截，回退 probe：{why}")
            self.registry.stage(cand, idx, decision.action.value)
            self.registry.promote(score=gate_dict.get("candidate_score", float("nan")))
            self._result.replay_pairs.append(
                (gate_dict.get("online_replay", float("nan")),
                 gate_dict.get("candidate_replay", float("nan")))
            )
            self.err_monitor.reset()
            return decision, gate_dict, experiments, used

        if scored:
            # 全部被拒：计一次失败，达到上限则回滚
            _, decision, cand, gate_dict = scored[0]
            self.registry.stage(cand, idx, decision.action.value)
            self.registry.discard()
            limit = self.cfg.gate.get("max_consecutive_failures", 3)
            if self.registry.consecutive_failures >= limit:
                if self.registry.rollback_to_best(idx) is not None:
                    self.err_monitor.reset()
            return decision, gate_dict, experiments, used

        return fallback, {}, experiments, used

    def _judge(self, cand, val_X, val_y):
        """只做门控判定，不改变 registry 状态。

        特征子集模型内部自带 feature_mask，predict 时自动套用，因此调用方
        统一传全维 val_X，与其他候选在同一数据上公平比较。
        """
        from .gate.evaluator import GateResult

        if val_y.size == 0:
            return GateResult(accepted=False, reason="验证集为空，无法判定")
        return self.gate.evaluate(
            cand, self.registry.online, self.baseline, (val_X, val_y), self.replay
        )

    # ------------------------------------------------------------------
    def _record(self, idx, report, decision, history_read, experiments,
                gate_dict, perf, seconds) -> None:
        """写审计记录，并给上一批回填实际结果。"""
        rec = DecisionRecord(
            batch_index=idx,
            trigger=report.notes[0] if report.notes else report.drift_type,
            drift_type=report.drift_type,
            evidence=report.to_dict(),
            history_read=history_read,
            hypothesis=self._hypothesis(report, decision),
            experiments=experiments,
            decision=decision.to_dict(),
            gate_result=gate_dict,
            model_version=(
                self.registry.online_version.version
                if self.registry.online_version else None
            ),
            cost={
                "action_cost": ACTION_COST[decision.action],
                "trainings_total": self.guard.total_trainings,
                "seconds": seconds,
                "tokens": 0,          # 第二阶段接入大模型后填写
            },
        )
        self.audit.append(rec)
        # 本批的真实表现就是上一批决策的结果，回填供复盘
        if idx > 0:
            self.audit.backfill_outcome(
                idx - 1,
                {"next_batch_score": perf.score, "next_batch_drift": report.drift_type},
            )
        self.swan.log(
            {
                self.metric: perf.value,
                "score": perf.score,
                "accuracy": perf.accuracy,
                "macro_f1": perf.macro_f1,
                "critical_recall": perf.critical_recall,
                "rmse": perf.rmse,
                "psi_mean": report.dist.psi_mean,
                "action_cost": ACTION_COST[decision.action],
                "gate_accepted": int(bool(gate_dict.get("accepted", False))),
            },
            step=idx,
        )

    def _scaler_snapshot(self):
        """给新训练的模型一份当前标准化参数的冻结副本。

        模型必须带着训练时的量纲去推理。第一阶段批次 2 出现负收益，就是因为
        推理时用的是"又多见了一批数据"之后的统计量，与训练时不一致。
        """
        if not self.cfg.data.get("standardize"):
            return None
        return self.scaler.copy()

    def _reference_score(self, current: float) -> float:
        """参照线取"历史上达到过的最佳批次成绩"。

        三次迭代后定在这个口径上，理由：
        - 近期滑动均值：缓慢退化时参照跟着一起下滑，漂移永远检测不出来
        - 当前版本历史最佳：换模型后参照被重置，失败的上线不会被标记
        - 门控成绩播种：门控分来自验证切片，与批次级成绩不同量级，会把
          "验证集到下一批"的正常落差误判成剧烈退化

        - 全局历史最佳：门槛过高，频繁触发重训反而拉低综合性能（实测 0.5881）

        最终用"当前版本历史最佳"，语义是"这个模型部署后达到过的水平"。
        它原本的缺陷（失败的上线不会被标记）已由门控改用留出验证集化解：
        候选不再能用自己训练过的数据刷高分，坏模型基本进不来。
        """
        ver = (
            self.registry.online_version.version
            if self.registry.online_version else 0
        )
        seen = self._version_scores.get(ver, [])
        return max(seen) if seen else current

    def _note_score(self, value: float, version: int) -> None:
        """把成绩记到产生它的那个模型版本上。

        必须显式传版本：本批成绩由"更新前"的在线模型产生，
        而调用点在门控上线之后，此时 registry 里已是新版本。
        """
        if value != value:
            return
        self._version_scores.setdefault(version, []).append(value)
        self._recent_scores.append(value)

    @staticmethod
    def _hypothesis(report, decision) -> str | None:
        """规则版由判定依据合成假设；大模型版直接用其输出。

        标注 [规则推导] 以便与第二阶段大模型提出的假设区分，不混淆来源。
        """
        if decision.hypothesis:
            return decision.hypothesis
        if decision.action is Action.NOOP:
            return None
        return (
            f"[规则推导] 判定为 {report.drift_type}，"
            f"性能变化 {report.perf_delta:+.4f}，"
            f"预期 {decision.action.value} 可恢复性能且不损害历史工况"
        )

    def _row(self, idx, y_true, pred, decision, gate_dict, cost, seconds,
             report=None) -> dict[str, Any]:
        perf = (
            evaluate(y_true, pred, self.class_set, self.metric, self.task_type,
                     self.critical_class)
            if pred is not None else None
        )
        return {
            "batch": idx,
            "n_samples": int(len(y_true)),
            # score：统一口径，越大越好（回归为 -RMSE），供 metrics 汇总
            "score": perf.score if perf else float("nan"),
            # value：原始主指标（分类 Macro-F1 / 回归 RMSE），供人读
            "value": perf.value if perf else float("nan"),
            "accuracy": perf.accuracy if perf else float("nan"),
            "macro_f1": perf.macro_f1 if perf else float("nan"),
            "critical_recall": perf.critical_recall if perf else float("nan"),
            "rmse": perf.rmse if perf else float("nan"),
            "asym": perf.asym if perf else float("nan"),
            "drift_type": report.drift_type if report else "",
            "psi_mean": report.dist.psi_mean if report else float("nan"),
            "action": decision.action.value,
            "gate_accepted": bool(gate_dict.get("accepted", False)),
            "model_version": (
                self.registry.online_version.version
                if self.registry.online_version else None
            ),
            "action_cost": cost,
            "seconds": seconds,
        }

    def _fail(self, idx, stage, exc) -> None:
        self._result.failures.append(
            {"batch_index": idx, "stage": stage,
             "error": f"{type(exc).__name__}: {exc}"}
        )


def _dist_kwargs(cfg) -> dict:
    m = cfg.monitor
    return {
        "psi_bins": m.get("psi_bins", 10),
        "psi_threshold": m.get("psi_threshold", 0.25),
        "psi_feature_ratio": m.get("psi_feature_ratio", 0.15),
        "ks_alpha": m.get("ks_alpha", 0.05),
        "pca_components": m.get("pca_components", 8),
        "pca_threshold": m.get("pca_threshold", 2.0),
    }
