"""10 轮官方续训的编排闭环。

每轮：读历史 → 决策（超参/是否训）→ 调官方续训 → 读官方验证 → 门控（上线/不上线）
→ 记录（审计四要素 + SwanLab）。门控未过则下一轮从上一个被采纳 checkpoint 继续。
全程无人干预；子进程失败/超时降级不中断。
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from harness.record.schema import DecisionRecord
from harness.record.audit import AuditLog
from harness.record.swan import SwanRecorder

from .gate import GateDecision, OfficialGate
from .metric import round_value, summarize_rounds, to_score
from .migrate import _load_official_model_cls
from .package import OfficialPackage
from .policy import RulePolicy, expand_candidates
from .significance import derive_seed
from .trainer import run_round


@dataclass
class RoundRow:
    round: int
    value: float
    accepted: bool
    best_value: float
    seconds: float
    action: str


@dataclass
class DriverResult:
    rows: list[RoundRow] = field(default_factory=list)
    baseline_value: float = float("nan")
    best_value: float = float("nan")
    promotions: int = 0
    trainings: int = 0
    skipped: int = 0
    failures: list[dict] = field(default_factory=list)
    best_checkpoint: Path | None = None
    seconds: float = 0.0
    # 双轨：显著性确认的那条链（最终交付优先取它，不存在则回退 best_*）
    best_significant_value: float = float("nan")
    best_significant_checkpoint: Path | None = None
    significant_promotions: int = 0
    rejected_by_verdict: list[int] = field(default_factory=list)
    # LLM 慢通路降级轮数（与训练失败 failures 分开统计，避免「LLM 从未生效」被掩盖）
    llm_degraded_rounds: int = 0
    # 实际调用 LLM 的轮数（hybrid 下只有停滞轮才调用）
    llm_calls: int = 0

    @property
    def deliverable_checkpoint(self) -> Path | None:
        """最终交付：优先显著性确认的 checkpoint，否则回退验证最优（保证不劣于现状）。"""
        return self.best_significant_checkpoint or self.best_checkpoint

    @property
    def deliverable_value(self) -> float:
        if self.best_significant_checkpoint is not None:
            return self.best_significant_value
        return self.best_value


def _baseline_value(pkg: OfficialPackage) -> float:
    """官方随包 training.json 的 validation_after 主指标，作为门槛基线与初始最优。"""
    report = json.loads((pkg.root / "training.json").read_text())
    return round_value(pkg, report.get("validation_after", {}))


def load_verdicts(path: str | Path | None) -> dict:
    """读平台侧裁决文件（可选）。缺失或损坏时返回空 dict，不中断闭环。"""
    if not path:
        return {}
    try:
        data = json.loads(Path(path).read_text())
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


class OfficialDriver:
    def __init__(self, pkg: OfficialPackage, rounds_dir: Path, *, rounds: int = 10,
                 min_gain: float = 0.0, policy: RulePolicy | None = None,
                 audit_path: str | Path = "outputs/official_audit.jsonl",
                 swan: SwanRecorder | None = None, timeout_s: float = 600.0,
                 candidates: int = 1, start_checkpoint: Path | None = None,
                 profile_override: Path | None = None,
                 entry_override: Path | None = None,
                 gate_mode: str = "plain", bootstrap: int = 500,
                 p_threshold: float = 0.95,
                 verdicts: dict | None = None) -> None:
        self.pkg = pkg
        self.rounds_dir = Path(rounds_dir)
        self.rounds = rounds
        self.candidates = max(1, int(candidates))
        # 加宽实验：从迁移后的 checkpoint 起跑，并用只改 hidden 的 profile
        self.start_checkpoint = Path(start_checkpoint) if start_checkpoint else None
        self.profile_override = Path(profile_override) if profile_override else None
        self.entry_override = Path(entry_override) if entry_override else None
        self.policy = policy or RulePolicy()
        self.audit = AuditLog(audit_path, reset=True)
        self.swan = swan or SwanRecorder(enabled=False)
        self.timeout_s = timeout_s
        self.baseline_value = _baseline_value(pkg)
        self.gate_mode = gate_mode
        self.gate = OfficialGate(pkg, self.baseline_value, min_gain=min_gain,
                                 mode=gate_mode, bootstrap=bootstrap,
                                 p_threshold=p_threshold)
        # 显著性模式下仍需一条普通链维持探索（续训起点用它）
        self._plain_gate = (self.gate if gate_mode == "plain"
                            else OfficialGate(pkg, self.baseline_value,
                                              min_gain=min_gain, mode="plain"))
        # 平台侧裁决：{"round-3": "rejected"}；被拒轮次不更新任何 best 链
        self.verdicts = dict(verdicts or {})
        self._val_cache: tuple | None = None
        self._pred_cache: dict[str, object] = {}

    def run(self) -> DriverResult:
        res = DriverResult(baseline_value=self.baseline_value,
                           best_value=self.baseline_value,
                           best_checkpoint=self.start_checkpoint)
        # start_checkpoint 为 None 时沿用随包 checkpoint（--from-baseline）
        if self.rounds_dir.exists():
            shutil.rmtree(self.rounds_dir)
        self.rounds_dir.mkdir(parents=True)
        history: list[dict] = []
        t0 = time.perf_counter()

        for k in range(1, self.rounds + 1):
            # 告知策略本轮将从哪个 checkpoint 续训（用于去重判定）
            if hasattr(self.policy, "next_parent"):
                self.policy.next_parent = str(res.best_checkpoint)
            plan = self.policy.decide(history)
            llm_tokens = int(getattr(self.policy, "last_tokens", 0) or 0)
            llm_degraded = getattr(self.policy, "last_degraded", None)
            if llm_degraded is not None:          # None = 本轮没走 LLM（规则/混合的快通路）
                res.llm_calls += 1
            if llm_degraded:
                res.llm_degraded_rounds += 1
            cands = expand_candidates(plan, self.candidates)
            t = time.perf_counter()
            # NOOP 轮：不训练、不计训练次数，但照常写审计（官方认可"不更新也是正确决策"）
            if plan.skip:
                res.rows.append(RoundRow(k, float("nan"), False, res.best_value,
                                         0.0, "NOOP"))
                history.append({"round": k, "value": float("nan"), "accepted": False,
                                "lr": plan.learning_rate, "epochs": plan.epochs,
                                "parent": str(res.best_checkpoint), "trained": False,
                                "reason": plan.reason or "NOOP"})
                self._record(k, plan, None, None, plan.reason,
                             llm_tokens, llm_degraded, [])
                self.swan.log({"skipped": 1}, step=k)
                res.skipped += 1
                continue
            rr, plan, cand_log = self._best_candidate(k, cands, res)
            seconds = time.perf_counter() - t
            if rr is None:
                self._record(k, plan, None, None, "本轮全部候选训练失败",
                             llm_tokens, llm_degraded, cand_log)
                history.append({"round": k, "value": float("nan"),
                                "accepted": False, "lr": plan.learning_rate,
                                "epochs": plan.epochs,
                                "parent": str(res.best_checkpoint), "trained": True,
                                "reason": "本轮全部候选训练失败"})
                continue

            value = round_value(self.pkg, rr.validation_after)
            parent_before = str(res.best_checkpoint)

            # 平台侧裁决优先：被拒的轮次不更新任何 best 链
            verdict = str(self.verdicts.get(f"round-{k}", "")).lower()
            if verdict in ("rejected", "reject", "false"):
                res.rejected_by_verdict.append(k)
                decision = GateDecision(False, f"平台未采纳本轮候选（verdict={verdict}）",
                                        float("nan"), float("nan"), float("nan"))
                action = "VERDICT_REJECT"
            else:
                decision = self._judge(k, value, rr, res)
                if decision.accepted:
                    res.best_value = value
                    res.best_checkpoint = rr.checkpoint
                    res.promotions += 1
                    action = "PROMOTE"
                else:
                    action = "REJECT"
                self._update_significant_track(k, value, rr, res)
            res.rows.append(RoundRow(k, value, decision.accepted,
                                     res.best_value, seconds, action))
            history.append({"round": k, "value": value,
                            "accepted": decision.accepted, "lr": plan.learning_rate,
                            "epochs": plan.epochs, "parent": parent_before,
                            "trained": True, "reason": decision.reason})
            self._record(k, plan, rr, decision, "", llm_tokens, llm_degraded, cand_log)
            # 记录显著性时一律用安全取值：日志绝不能把闭环搞挂
            sig_payload = (getattr(self, "_last_significance", None) or {}).get("significance")
            extra = {}
            if isinstance(sig_payload, dict) and "p_better" in sig_payload:
                extra = {"p_better": sig_payload["p_better"],
                         "best_significant": res.best_significant_value}
            # 本轮实际用的超参与相对基线增益也上报，平台上能直接看出「改了什么→结果如何」
            gain = to_score(self.pkg, res.best_value) - to_score(self.pkg, self.baseline_value)
            if plan.learning_rate is not None:
                extra["learning_rate"] = plan.learning_rate
            self.swan.log(
                {self.pkg.metric_name: value, "best": res.best_value,
                 "baseline": self.baseline_value, "gain_vs_baseline": gain,
                 "epochs": plan.epochs, "accepted": int(decision.accepted),
                 "llm_tokens": llm_tokens, "llm_degraded": int(bool(llm_degraded)),
                 "candidates": len(cand_log), **extra}, step=k,
            )

        res.seconds = time.perf_counter() - t0
        self.swan.finish()
        return res

    def _validation(self):
        """懒加载官方验证集（x, y），供显著性检验用。只读一次。"""
        if self._val_cache is None:
            import sys as _sys
            _sys.path.insert(0, str(self.pkg.root))
            from baselines.data import read_dataset
            cfg = json.loads(self.pkg.config_path().read_text())
            classes = cfg.get("classes", [])
            tr = read_dataset(self.pkg.train_jsonl, self.pkg.task, classes)
            va = read_dataset(self.pkg.validation_jsonl, self.pkg.task, classes,
                              tr["paths"])
            self._val_cache = (va["x"], va["y"])
        return self._val_cache

    def _predict(self, checkpoint):
        """用官方 Model 在验证集上预测，按 checkpoint 路径缓存。失败返回 None。"""
        if checkpoint is None:
            return None
        key = str(checkpoint)
        if key in self._pred_cache:
            return self._pred_cache[key]
        try:
            Model = _load_official_model_cls(self.pkg.root)
            x, _ = self._validation()
            pred = Model.load(checkpoint).predict(x)
        except Exception:  # noqa: BLE001 —— 显著性是增强项，失败即降级
            pred = None
        self._pred_cache[key] = pred
        return pred

    def _judge(self, k, value, rr, res):
        """普通链的门控判定（续训起点用这条链，保持探索能力）。"""
        return self._plain_gate.evaluate(value, res.best_value)

    def _update_significant_track(self, k, value, rr, res) -> None:
        """显著性链：只有统计显著优于上一个显著模型时才更新。"""
        if self.gate_mode != "significance":
            return
        ref_ckpt = res.best_significant_checkpoint or self.start_checkpoint \
            or self.pkg.checkpoint
        ref_value = (res.best_significant_value
                     if res.best_significant_checkpoint is not None
                     else self.baseline_value)
        try:
            _, y = self._validation()
        except Exception:  # noqa: BLE001
            return
        sig_decision = self.gate.evaluate(
            value, ref_value,
            pred_cand=self._predict(rr.checkpoint),
            pred_best=self._predict(ref_ckpt), y=y,
            seed=derive_seed(self.pkg.parameter_count or 42, k),
            cand_steps=(rr.report or {}).get("final_steps"),
            best_steps=None,
        )
        self._last_significance = sig_decision.to_dict()
        if sig_decision.accepted:
            res.best_significant_value = value
            res.best_significant_checkpoint = rr.checkpoint
            res.significant_promotions += 1

    def _best_candidate(self, k, cands, res):
        """本轮对每个候选各训一次（均从当前被采纳 checkpoint 出发），取验证最优。

        返回 (最优 RoundResult 或 None, 对应 plan, 候选日志)。单候选时目录/stage_id
        与改动前一致（round-k），保证 K=1 行为不变。单个候选失败只跳过。
        """
        multi = len(cands) > 1
        best_rr, best_plan, best_score = None, cands[0], float("-inf")
        log: list[dict] = []
        for j, cp in enumerate(cands):
            name = f"round-{k}-cand-{j}" if multi else f"round-{k}"
            rr = run_round(
                self.pkg, self.rounds_dir / name, resume=res.best_checkpoint,
                epochs=cp.epochs, stage_id=name, learning_rate=cp.learning_rate,
                batch_size=cp.batch_size, timeout_s=self.timeout_s,
                profile_override=self.profile_override,
                entry_override=self.entry_override,
            )
            res.trainings += 1
            entry = {"epochs": cp.epochs, "learning_rate": cp.learning_rate,
                     "stage_id": name}
            if not rr.ok:
                res.failures.append({"round": k, "candidate": j, "error": rr.error})
                entry["error"] = rr.error
                log.append(entry)
                continue
            value = round_value(self.pkg, rr.validation_after)
            entry["value"] = value
            log.append(entry)
            score = to_score(self.pkg, value)
            if score == score and score > best_score:
                best_rr, best_plan, best_score = rr, cp, score
        for e in log:
            e["selected"] = best_rr is not None and e["stage_id"] == best_rr.stage_id
        return best_rr, best_plan, log

    def _record(self, k, plan, rr, decision, error, llm_tokens=0, llm_degraded=None,
                cand_log=None) -> None:
        rec = DecisionRecord(
            batch_index=k,
            trigger=f"round-{k}",
            drift_type="OFFICIAL_RETRAIN",
            evidence={"validation_after": (rr.validation_after if rr else {}),
                      "error": error},
            history_read=plan.history_read,
            hypothesis=plan.hypothesis,
            experiments=cand_log or [{"epochs": plan.epochs,
                                      "learning_rate": plan.learning_rate,
                                      "stage_id": f"round-{k}"}],
            decision={"action": "train", "accepted": bool(decision and decision.accepted),
                      "reason": decision.reason if decision else error,
                      "llm_degraded": llm_degraded},
            gate_result=(
                {"accepted": decision.accepted, "reason": decision.reason,
                 "candidate_score": decision.candidate_score,
                 "best_score": decision.best_score,
                 "baseline_score": decision.baseline_score,
                 # 显著性链的判定（显著性模式下才有），供审计追溯
                 "significance": getattr(self, "_last_significance", None)}
                if decision else {"accepted": False, "reason": error}
            ),
            cost={"seconds": rr and rr.report.get("training_seconds"), "tokens": llm_tokens},
        )
        self.audit.append(rec)
