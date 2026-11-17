"""每轮续训的决策策略。

规则版：固定每轮 epochs；若上一轮被拒（验证未提升），下一轮减半学习率再试，
并仍从上一个被采纳的 checkpoint 继续。预留 Decider 协议给第二阶段 LLM 慢通路
（读 history→提超参/改代码）。本层不在延迟关键路径上做大模型调用。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable

EPOCH_BOUNDS = (1, 50)
LR_BOUNDS = (1e-6, 0.1)


@dataclass
class RoundPlan:
    epochs: int
    learning_rate: float | None = None
    batch_size: int | None = None
    hypothesis: str = ""
    history_read: list[int] = field(default_factory=list)
    skip: bool = False          # 本轮判为 NOOP：不训练，但照常写审计
    reason: str = ""            # 跳过或换轴的理由


def expand_candidates(base: RoundPlan, k: int) -> list[RoundPlan]:
    """以 base 为中心生成最多 k 个候选（含 base 本身），裁剪到官方合法区间并去重。

    顺序固定（可复现）：base、lr×0.5、lr×2、epochs-2、epochs+2。
    lr 为 None（沿用 config）时只扰动 epochs，避免制造非法 lr。
    skip 的 plan 不扩候选（本轮本来就不训练）。
    """
    if k <= 1 or base.skip:
        return [base]
    raw: list[tuple[int, float | None, str]] = [(base.epochs, base.learning_rate, "中心")]
    if base.learning_rate is not None:
        raw.append((base.epochs, base.learning_rate * 0.5, "lr×0.5"))
        raw.append((base.epochs, base.learning_rate * 2.0, "lr×2"))
    raw.append((base.epochs - 2, base.learning_rate, "epochs-2"))
    raw.append((base.epochs + 2, base.learning_rate, "epochs+2"))

    out: list[RoundPlan] = []
    seen: set[tuple[int, float | None]] = set()
    for epochs, lr, tag in raw:
        epochs = max(EPOCH_BOUNDS[0], min(EPOCH_BOUNDS[1], epochs))
        if lr is not None:
            lr = max(LR_BOUNDS[0], min(LR_BOUNDS[1], lr))
        key = (epochs, None if lr is None else round(lr, 12))
        if key in seen:
            continue
        seen.add(key)
        out.append(replace(base, epochs=epochs, learning_rate=lr,
                           hypothesis=f"{base.hypothesis}（候选:{tag}）"))
        if len(out) >= k:
            break
    return out


@runtime_checkable
class Decider(Protocol):
    def decide(self, history: list[dict]) -> RoundPlan: ...


class RulePolicy:
    """history 每项：{"round","value","accepted","lr"}。"""

    def __init__(self, base_epochs: int = 5, base_lr: float | None = None,
                 lr_decay_on_reject: float = 0.5, min_lr: float = 1e-5) -> None:
        self.base_epochs = base_epochs
        self.base_lr = base_lr
        self.lr_decay_on_reject = lr_decay_on_reject
        self.min_lr = min_lr

    def decide(self, history: list[dict]) -> RoundPlan:
        read = [h["round"] for h in history[-5:]]
        if not history:
            return RoundPlan(self.base_epochs, self.base_lr,
                             hypothesis="首轮：按配置超参从官方基线续训",
                             history_read=read)
        last = history[-1]
        lr = last.get("lr", self.base_lr)
        if not last["accepted"]:
            # 上一轮没提升：学习率减半再试，继续从被采纳的 checkpoint 出发
            if lr is not None:
                lr = max(lr * self.lr_decay_on_reject, self.min_lr)
            return RoundPlan(
                self.base_epochs, lr,
                hypothesis=f"上一轮未提升（{last['value']:.4f}），降学习率至 {lr} 再试",
                history_read=read,
            )
        return RoundPlan(
            self.base_epochs, lr,
            hypothesis=f"上一轮已提升至 {last['value']:.4f}，保持学习率继续续训",
            history_read=read,
        )


class AdaptivePolicy:
    """停滞检测 + 多轴探索 + 无效轮跳过。

    现有 RulePolicy 只有一个轴：被拒就把 lr 减半。实测 A 包的 lr 会一路衰减到 1e-5
    量级，5 个 epoch 已经动不了模型，连续多轮验证值一模一样——纯消耗零信息。
    本策略：
      1. 停滞检测：上一轮验证值与前一轮差 < tol，判定 lr 进入无效区
      2. 换轴：此时 epochs ×2（上限 50）、lr 回弹到上次"仍有变化"的值，而不是继续减半
      3. 去重：(parent, epochs, lr) 已试过 → 返回 skip=True 的 NOOP，不浪费训练
      4. 常规：有提升则保持；被拒但仍有变化则按原逻辑减半 lr
      5. 兜底：轴耗尽则 NOOP；但第 1 轮永不跳过

    history 每项：{"round","value","accepted","lr","parent"(可选),"epochs"(可选),"trained"(可选)}
    """

    def __init__(self, base_epochs: int = 5, base_lr: float | None = None,
                 lr_decay_on_reject: float = 0.5, min_lr: float = 1e-5,
                 stall_tol: float = 1e-9, max_epochs: int = 50) -> None:
        self.base_epochs = base_epochs
        self.base_lr = base_lr
        self.lr_decay_on_reject = lr_decay_on_reject
        self.min_lr = min_lr
        self.stall_tol = stall_tol
        self.max_epochs = max_epochs
        self.tried: set[tuple] = set()          # 已试过的 (parent, epochs, lr)
        self._last_effective_lr: float | None = None
        # 闭环在每轮 decide 前写入"本轮将从哪个 checkpoint 续训"，用于去重判定
        self.next_parent: str | None = None

    @staticmethod
    def _key(parent, epochs, lr):
        return (str(parent), int(epochs), None if lr is None else round(float(lr), 12))

    def _clamp(self, epochs: int, lr: float | None):
        epochs = max(EPOCH_BOUNDS[0], min(self.max_epochs, int(epochs)))
        if lr is not None:
            lr = max(LR_BOUNDS[0], min(LR_BOUNDS[1], float(lr)))
        return epochs, lr

    def _trained(self, history):
        return [h for h in history if h.get("trained", True)
                and h.get("value") == h.get("value")]

    def _stalled(self, done) -> bool:
        """最近两次实际训练的验证值几乎一样 → lr 已无效。"""
        if len(done) < 2:
            return False
        return abs(done[-1]["value"] - done[-2]["value"]) < self.stall_tol

    def decide(self, history: list[dict]) -> RoundPlan:
        read = [h["round"] for h in history[-5:]]
        parent = self.next_parent
        done = self._trained(history)

        if not done:
            epochs, lr = self._clamp(self.base_epochs, self.base_lr)
            self._last_effective_lr = lr
            plan = RoundPlan(epochs, lr, hypothesis="首轮：按配置超参从官方基线续训",
                             history_read=read)
            self.tried.add(self._key(parent, epochs, lr))
            return plan

        last = done[-1]
        lr = last.get("lr", self.base_lr)
        epochs = int(last.get("epochs", self.base_epochs))

        if last["accepted"]:
            hypo = f"上一轮提升至 {last['value']:.4f}，保持超参继续续训"
            self._last_effective_lr = lr
        elif self._stalled(done):
            # 换轴：lr 已无效，改为加大 epochs，并把 lr 回弹到上次有效值
            lr = self._last_effective_lr if self._last_effective_lr is not None else lr
            epochs = epochs * 2
            hypo = (f"连续两轮验证值无变化（{last['value']:.4f}），判定学习率进入无效区："
                    f"换轴改为 epochs→{min(epochs, self.max_epochs)}，lr 回弹到 {lr}")
        else:
            if lr is not None:
                lr = max(lr * self.lr_decay_on_reject, self.min_lr)
            hypo = f"上一轮未提升（{last['value']:.4f}），降学习率至 {lr} 再试"

        epochs, lr = self._clamp(epochs, lr)
        key = self._key(parent, epochs, lr)
        if key in self.tried:
            return RoundPlan(epochs, lr, hypothesis=f"{hypo}；但该配置已试过",
                             history_read=read, skip=True,
                             reason=f"(parent,epochs={epochs},lr={lr}) 已试过，判 NOOP 省算力")
        self.tried.add(key)
        return RoundPlan(epochs, lr, hypothesis=hypo, history_read=read)
