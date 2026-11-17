"""PyTRiO 大模型慢通路：读实验历史 → 提下一轮续训超参。

离线安全：大模型只影响"下一轮续训超参"的离线决策，不在延迟关键路径；任何失败
（无端点/网络/超时/坏 JSON/越界）都降级到规则快通路且不抛错。只用平台千问
（OpenAI 兼容），base_url/model/key 来自环境变量；缺任一即视为不可用。
"""

from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.request

from .policy import RoundPlan, RulePolicy

try:  # macOS/Python.framework 常缺系统 CA，用 certifi 的证书做 TLS 校验
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:  # pragma: no cover —— 无 certifi 时回退系统默认
    _SSL_CTX = ssl.create_default_context()

EPOCH_RANGE = (1, 50)
LR_RANGE = (1e-6, 0.1)
_RAW_CAP = 2000


class LLMClient:
    """最小 OpenAI 兼容客户端（chat.completions）。一次调用、短超时、不重试。"""

    def __init__(self, base_url: str | None = None, model: str | None = None,
                 api_key: str | None = None, timeout_s: float | None = None) -> None:
        self.base_url = (base_url or os.environ.get("PYTRIO_BASE_URL", "")).rstrip("/")
        self.model = model or os.environ.get("PYTRIO_MODEL", "")
        self.api_key = api_key or os.environ.get("PYTRIO_API_KEY", "")
        if timeout_s is None:
            timeout_s = float(os.environ.get("PYTRIO_TIMEOUT_S", "300"))
        self.timeout_s = timeout_s

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.model and self.api_key)

    def complete(self, system: str, user: str) -> tuple[str, int]:
        """返回 (文本, token 数)。不可用或出错时抛异常，由上层捕获降级。"""
        if not self.available:
            raise RuntimeError("PyTRiO 端点未配置（PYTRIO_BASE_URL/MODEL/API_KEY）")
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            # temperature=0 尽量确定（C11 可复现）；远端模型仍不保证逐字一致
            "temperature": 0, "max_tokens": 512,
            "enable_thinking": False,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=payload,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s, context=_SSL_CTX) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        msg = body["choices"][0]["message"]
        text = msg.get("content") or msg.get("reasoning") or ""
        if not text.strip():
            raise ValueError("PyTRiO 返回空文本")
        tokens = int(body.get("usage", {}).get("total_tokens", 0))
        return text, tokens


def _clamp(value, low, high):
    return max(low, min(high, value))


def _extract_json(text: str) -> dict:
    """从回复里取第一个 JSON 对象（容忍包裹的 ```json 或解释文字）。"""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("回复中没有 JSON 对象")
    return json.loads(text[start:end + 1])


class LLMDecider:
    """实现 policy.Decider：读历史让千问提下一轮续训超参，失败降级规则策略。"""

    def __init__(self, profile: str, task: str, metric_name: str,
                 higher_better: bool, baseline_value: float,
                 fallback: RulePolicy, client: LLMClient | None = None) -> None:
        self.profile = profile
        self.task = task
        self.metric_name = metric_name
        self.higher_better = higher_better
        self.baseline_value = baseline_value
        self.fallback = fallback
        self.client = client or LLMClient()
        self.last_tokens = 0
        self.last_degraded = False
        self.last_raw = ""

    @staticmethod
    def _cfg_key(epochs, lr) -> tuple[int, float]:
        return int(epochs), round(float(lr), 10)

    def _tried(self, history: list[dict]) -> set[tuple[int, float]]:
        """已实际训练过的 (epochs, lr)。用于提示「禁止重复」并拦截重复提案。"""
        return {self._cfg_key(h["epochs"], h["lr"]) for h in history
                if h.get("trained", True) and h.get("epochs") is not None
                and h.get("lr") is not None}

    def _best(self, history: list[dict]) -> float:
        """当前已上线最优：被采纳轮次中的最好值，没有则为官方基线。"""
        ok = [h["value"] for h in history if h.get("accepted") and h["value"] == h["value"]]
        if not ok:
            return self.baseline_value
        return max(ok) if self.higher_better else min(ok)

    def _prompt(self, history: list[dict]) -> tuple[str, str]:
        direction = "越大越好" if self.higher_better else "越小越好"
        lines = []
        for h in history[-8:]:
            if h["value"] != h["value"]:
                continue
            status = "已上线" if h["accepted"] else "未采纳"
            why = f"（{h['reason']}）" if h.get("reason") and not h["accepted"] else ""
            lines.append(f"  轮{h['round']}: epochs={h.get('epochs')} lr={h.get('lr')} → "
                         f"{self.metric_name}={h['value']:.4f} {status}{why}")
        hist = "\n".join(lines) if lines else "  （暂无历史）"
        tried = sorted(self._tried(history))
        tried_txt = "、".join(f"(epochs={e}, lr={lr:g})" for e, lr in tried) or "无"
        base_ep = getattr(self.fallback, "base_epochs", None)
        base_lr = getattr(self.fallback, "base_lr", None)
        system = (
            "你是工业模型续训的调参助手。模型是固定结构的 CPU NumPy MLP，每轮都从当前已上线的"
            " checkpoint 继续训练（微调），只能调 epochs 和 learning_rate，不能改网络结构或特征。"
            "门控：验证指标不优于官方基线或当前已上线最优的候选不会上线。"
            "请结合历史中 epochs、lr 与结果的关系提出一个可实验的原子改动；"
            "不得重复已试过的配置。只输出 JSON。"
        )
        user = (
            f"任务 {self.profile}（{self.task}），主指标 {self.metric_name}（{direction}）。\n"
            f"官方基线 {self.baseline_value:.4f}；当前已上线最优 {self._best(history):.4f}。\n"
            f"官方默认续训配置（已知安全起点）：epochs={base_ep}, learning_rate={base_lr}。\n"
            f"历史：\n{hist}\n"
            f"已试过的配置（禁止重复）：{tried_txt}\n\n"
            "给出下一轮续训的超参，严格输出 JSON："
            '{"epochs": 整数1-50, "learning_rate": 0到0.1之间小数, "hypothesis": "一句话理由"}'
        )
        return system, user

    def decide(self, history: list[dict]) -> RoundPlan:
        self.last_tokens, self.last_degraded, self.last_raw = 0, False, ""
        read = [h["round"] for h in history[-5:]]
        try:
            system, user = self._prompt(history)
            text, tokens = self.client.complete(system, user)
            self.last_tokens, self.last_raw = tokens, text[:_RAW_CAP]
            data = _extract_json(text)
            epochs = int(data["epochs"])
            lr = float(data["learning_rate"])
            if not (isinstance(data.get("hypothesis"), str) and data["hypothesis"].strip()):
                raise ValueError("缺少 hypothesis")
            epochs = int(_clamp(epochs, *EPOCH_RANGE))
            lr = float(_clamp(lr, *LR_RANGE))
            if self._cfg_key(epochs, lr) in self._tried(history):
                raise ValueError(f"重复已试过的配置 epochs={epochs} lr={lr:g}")
            return RoundPlan(epochs=epochs, learning_rate=lr,
                             hypothesis=f"[LLM] {data['hypothesis'].strip()}",
                             history_read=read)
        except Exception as exc:  # noqa: BLE001 —— 慢通路绝不中断闭环
            self.last_degraded = True
            plan = self.fallback.decide(history)
            plan.hypothesis = f"[LLM降级:{type(exc).__name__}] {plan.hypothesis}"
            return plan


class HybridDecider:
    """快慢通路混合：常态走规则快通路，快通路不灵时才唤醒 LLM 慢通路。

    两个唤醒条件（任一满足）：
      1. 停滞：最近 patience 轮都没上分（未采纳 / NOOP / 训练失败）。
      2. 爬坡放缓：最近 slow_window 轮里，已上线最优的累计提升不足官方基线的 min_rel_gain。
         快通路每轮都有一点点提升、从不连续被拒时（实测 C 包），只看条件 1 永远唤不醒 LLM。
    首轮永远走快通路（官方默认配置是已知安全起点）；快通路仍在大步上分时不调用大模型，
    省 token，也避免它在顺风时添乱。对闭环暴露与 LLMDecider 相同的
    last_tokens / last_degraded / baseline_value，并把 next_parent 转交给快通路。

    熔断：连续 llm_patience 次 LLM 提案都没被采纳，就暂停调用 LLM、回到快通路；
    之后任何一轮有新上线（不论哪条通路）就重新启用。实测 27B 有用的提案都是每次停滞后的
    第一次调用，之后连着调用没有一次被采纳（A 包 7 次全部未采纳），熔断只省这部分开销。
    llm_patience=0 关闭熔断。
    """

    def __init__(self, rule, llm: LLMDecider, patience: int = 2,
                 slow_window: int = 2, min_rel_gain: float = 0.005,
                 llm_patience: int = 2) -> None:
        self.rule = rule
        self.llm = llm
        self.patience = max(1, int(patience))
        self.slow_window = max(1, int(slow_window))
        self.min_rel_gain = float(min_rel_gain)
        self.llm_patience = max(0, int(llm_patience))
        self.llm_misses = 0                       # 连续未被采纳的 LLM 提案数
        self.next_parent: str | None = None
        self.last_tokens = 0
        self.last_degraded: bool | None = None
        self.last_path = "rule"
        self.last_wake = ""

    @property
    def baseline_value(self) -> float:
        return self.llm.baseline_value

    @baseline_value.setter
    def baseline_value(self, value: float) -> None:
        self.llm.baseline_value = value

    def _stalled(self, history: list[dict]) -> bool:
        # NOOP / 训练失败也算「没上分」：快通路轴耗尽只会一直 NOOP，此时正该唤醒 LLM
        recent = history[-self.patience:]
        return len(recent) >= self.patience and not any(h["accepted"] for h in recent)

    def _recent_rel_gain(self, history: list[dict]) -> float | None:
        """最近 slow_window 轮已上线最优的累计提升，占官方基线绝对值的比例。轮数不足返回 None。"""
        if len(history) < self.slow_window:
            return None
        best = self.baseline_value
        traj = [best]
        for h in history:
            v = h["value"]
            if h.get("accepted") and v == v:
                best = v
            traj.append(best)
        now, before = traj[-1], traj[-1 - self.slow_window]
        gain = (now - before) if self.llm.higher_better else (before - now)
        return gain / max(abs(self.baseline_value), 1e-12)

    def _wake_reason(self, history: list[dict]) -> str:
        if self._stalled(history):
            return f"停滞：连续 {self.patience} 轮没上分"
        if self.min_rel_gain > 0:
            rel = self._recent_rel_gain(history)
            if rel is not None and rel < self.min_rel_gain:
                return (f"爬坡放缓：近 {self.slow_window} 轮提升 {rel:.2%} "
                        f"< {self.min_rel_gain:.2%}")
        return ""

    def decide(self, history: list[dict]) -> RoundPlan:
        # 先用上一轮结果更新熔断计数：有上线就重置；上一轮是 LLM 提案却没上线就 +1
        if history:
            if history[-1].get("accepted"):
                self.llm_misses = 0
            elif self.last_path == "llm":
                self.llm_misses += 1
        self.last_wake = self._wake_reason(history)
        tripped = self.llm_patience > 0 and self.llm_misses >= self.llm_patience
        if self.last_wake and not tripped:
            plan = self.llm.decide(history)
            self.last_tokens, self.last_degraded = self.llm.last_tokens, self.llm.last_degraded
            self.last_path = "llm"
            plan.hypothesis = f"{plan.hypothesis}（唤醒原因：{self.last_wake}）"
            return plan
        if hasattr(self.rule, "next_parent"):
            self.rule.next_parent = self.next_parent
        plan = self.rule.decide(history)
        self.last_tokens, self.last_degraded, self.last_path = 0, None, "rule"
        note = (f"（LLM 熔断：连续 {self.llm_misses} 次提案未采纳，{self.last_wake}，暂用快通路）"
                if self.last_wake and tripped else "")
        plan.hypothesis = f"[快通路] {plan.hypothesis}{note}"
        return plan

