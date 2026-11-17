import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from driver.llm import HybridDecider, LLMClient, LLMDecider, _extract_json
from driver.policy import RulePolicy


class FakeClient:
    """可控的 LLM 客户端桩。"""

    def __init__(self, text=None, tokens=42, available=True, raises=None):
        self._text, self._tokens = text, tokens
        self.available = available
        self._raises = raises

    def complete(self, system, user):
        if self._raises is not None:
            raise self._raises
        return self._text, self._tokens


def decider(client, baseline=0.71):
    return LLMDecider(profile="A", task="classification", metric_name="macro_f1",
                      higher_better=True, baseline_value=baseline,
                      fallback=RulePolicy(base_epochs=5, base_lr=0.003), client=client)


HIST = [{"round": 1, "value": 0.72, "accepted": True, "lr": 0.003}]


# ---------- 客户端可用性 ----------

def test_client_unavailable_without_env(monkeypatch):
    for k in ("PYTRIO_BASE_URL", "PYTRIO_MODEL", "PYTRIO_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert not LLMClient().available


def test_client_available_with_env(monkeypatch):
    monkeypatch.setenv("PYTRIO_BASE_URL", "http://x/v1")
    monkeypatch.setenv("PYTRIO_MODEL", "qwen")
    monkeypatch.setenv("PYTRIO_API_KEY", "k")
    assert LLMClient().available


def test_client_default_timeout_without_env(monkeypatch):
    monkeypatch.delenv("PYTRIO_TIMEOUT_S", raising=False)
    assert LLMClient().timeout_s == 300.0


def test_client_reads_timeout_env(monkeypatch):
    monkeypatch.setenv("PYTRIO_TIMEOUT_S", "123.5")
    assert LLMClient().timeout_s == 123.5


# ---------- JSON 抽取 ----------

def test_extract_json_from_wrapped_text():
    assert _extract_json('噪声```json\n{"epochs": 3}\n```尾部')["epochs"] == 3


def test_extract_json_rejects_no_object():
    with pytest.raises(ValueError):
        _extract_json("没有大括号")


# ---------- 决策：正常路径 ----------

def test_decide_parses_and_builds_prompt_with_history_and_baseline():
    captured = {}

    class C:
        available = True

        def complete(self, system, user):
            captured["user"] = user
            return '{"epochs": 7, "learning_rate": 0.01, "hypothesis": "再训几轮"}', 55

    d = decider(C())
    plan = d.decide(HIST)
    assert plan.epochs == 7 and plan.learning_rate == 0.01
    assert plan.hypothesis.startswith("[LLM]")
    assert d.last_tokens == 55 and not d.last_degraded
    assert "macro_f1" in captured["user"] and "0.71" in captured["user"]  # 含基线
    assert "轮1" in captured["user"]                                      # 含历史


def test_decide_clamps_out_of_range():
    d = decider(FakeClient('{"epochs": 999, "learning_rate": 5.0, "hypothesis": "x"}'))
    plan = d.decide(HIST)
    assert plan.epochs == 50 and plan.learning_rate == 0.1       # 裁剪到合法区间
    assert not d.last_degraded


# ---------- 决策：降级路径（永不抛错） ----------

def test_decide_degrades_on_bad_json():
    d = decider(FakeClient("这不是 JSON"))
    plan = d.decide(HIST)
    assert d.last_degraded and plan.hypothesis.startswith("[LLM降级:")
    assert plan.epochs == 5                                       # 回退规则


def test_decide_degrades_on_missing_field():
    d = decider(FakeClient('{"epochs": 5}'))   # 缺 learning_rate/hypothesis
    plan = d.decide(HIST)
    assert d.last_degraded


def test_decide_degrades_when_unavailable():
    d = decider(FakeClient(available=True, raises=RuntimeError("端点未配置")))
    plan = d.decide(HIST)
    assert d.last_degraded and "[LLM降级:RuntimeError]" in plan.hypothesis


def test_decide_degrades_on_timeout():
    d = decider(FakeClient(raises=TimeoutError("超时")))
    plan = d.decide(HIST)
    assert d.last_degraded and plan.learning_rate == 0.003        # 规则兜底


def test_decide_never_raises_on_empty_history():
    d = decider(FakeClient('{"epochs": 3, "learning_rate": 0.002, "hypothesis": "首轮"}'))
    plan = d.decide([])
    assert plan.epochs == 3


# ---------- 提示词：锚点 + 门控反馈 + 已试配置 ----------

HIST_FULL = [
    {"round": 1, "value": 0.66, "accepted": False, "lr": 0.01, "epochs": 20,
     "trained": True, "reason": "候选 0.6600 低于官方基线"},
    {"round": 2, "value": 0.72, "accepted": True, "lr": 0.005, "epochs": 20,
     "trained": True, "reason": "通过"},
]


def test_prompt_carries_default_config_epochs_reason_and_tried():
    captured = {}

    class C:
        available = True

        def complete(self, system, user):
            captured["user"] = user
            return '{"epochs": 5, "learning_rate": 0.003, "hypothesis": "回到默认配置"}', 9

    decider(C()).decide(HIST_FULL)
    u = captured["user"]
    assert "epochs=5, learning_rate=0.003" in u              # 官方默认配置锚点
    assert "epochs=20 lr=0.01" in u                          # 历史含 epochs
    assert "低于官方基线" in u                                 # 未采纳原因回传
    assert "(epochs=20, lr=0.01)" in u and "(epochs=20, lr=0.005)" in u   # 禁止重复清单
    assert "当前已上线最优 0.7200" in u


def test_repeat_proposal_falls_back_to_rule():
    d = decider(FakeClient('{"epochs": 20, "learning_rate": 0.005, "hypothesis": "再来一次"}'))
    plan = d.decide(HIST_FULL)
    assert d.last_degraded
    assert plan.hypothesis.startswith("[LLM降级:ValueError]")
    assert (plan.epochs, plan.learning_rate) != (20, 0.005)


# ---------- 快慢混合 ----------

class CountingClient(FakeClient):
    def __init__(self, text):
        super().__init__(text)
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        return super().complete(system, user)


def hybrid(client, patience=2):
    llm = decider(client)
    return HybridDecider(RulePolicy(base_epochs=5, base_lr=0.003), llm, patience=patience)


def test_hybrid_uses_rule_until_stalled():
    c = CountingClient('{"epochs": 8, "learning_rate": 0.001, "hypothesis": "换轴"}')
    h = hybrid(c)
    plan = h.decide([])                                                   # 首轮
    assert c.calls == 0 and plan.hypothesis.startswith("[快通路]")
    assert h.last_degraded is None and h.last_tokens == 0
    one_reject = [{"round": 1, "value": 0.70, "accepted": False, "lr": 0.003, "epochs": 5}]
    h.decide(one_reject)                                                  # 只拒 1 轮，未停滞
    assert c.calls == 0


def test_hybrid_wakes_llm_after_patience_rejections():
    c = CountingClient('{"epochs": 8, "learning_rate": 0.001, "hypothesis": "换轴"}')
    h = hybrid(c)
    stalled = [{"round": 1, "value": 0.70, "accepted": False, "lr": 0.003, "epochs": 5},
               {"round": 2, "value": float("nan"), "accepted": False, "lr": 0.003,
                "epochs": 5, "trained": False}]                          # 拒绝 + NOOP
    plan = h.decide(stalled)
    assert c.calls == 1 and plan.hypothesis.startswith("[LLM]")
    assert h.last_degraded is False and h.last_tokens == 42
    h.baseline_value = 0.5
    assert h.llm.baseline_value == 0.5                                    # 基线透传给 LLM


def test_hybrid_wakes_llm_on_slow_climb():
    """每轮都被采纳但提升很小（C 包的慢速爬坡），停滞条件永远不满足，也要唤醒 LLM。"""
    c = CountingClient('{"epochs": 15, "learning_rate": 0.0005, "hypothesis": "加大训练"}')
    h = hybrid(c)                                         # 基线 0.71，阈值 0.5%
    creeping = [{"round": 1, "value": 0.712, "accepted": True, "lr": 0.003, "epochs": 5},
                {"round": 2, "value": 0.713, "accepted": True, "lr": 0.003, "epochs": 5}]
    plan = h.decide(creeping)                             # 两轮共 +0.003 ≈ 0.42% < 0.5%
    assert c.calls == 1 and "爬坡放缓" in plan.hypothesis


def test_hybrid_keeps_rule_while_climbing_fast():
    c = CountingClient('{"epochs": 15, "learning_rate": 0.0005, "hypothesis": "加大训练"}')
    h = hybrid(c)
    climbing = [{"round": 1, "value": 0.75, "accepted": True, "lr": 0.003, "epochs": 5},
                {"round": 2, "value": 0.76, "accepted": True, "lr": 0.003, "epochs": 5}]
    plan = h.decide(climbing)                             # 两轮共 +0.05 ≈ 7%
    assert c.calls == 0 and plan.hypothesis.startswith("[快通路]")
    h_off = HybridDecider(RulePolicy(base_epochs=5, base_lr=0.003), decider(c),
                          min_rel_gain=0)                 # 0 = 关闭爬坡触发
    h_off.decide([{"round": 1, "value": 0.712, "accepted": True, "lr": 0.003, "epochs": 5},
                  {"round": 2, "value": 0.713, "accepted": True, "lr": 0.003, "epochs": 5}])
    assert c.calls == 0


def test_hybrid_breaker_trips_after_llm_misses_and_rearms_on_promotion():
    c = CountingClient('{"epochs": 8, "learning_rate": 0.001, "hypothesis": "换轴"}')
    h = hybrid(c)                                         # llm_patience 默认 2
    hist = [{"round": 1, "value": 0.70, "accepted": False, "lr": 0.003, "epochs": 5},
            {"round": 2, "value": 0.70, "accepted": False, "lr": 0.0015, "epochs": 5}]
    h.decide(hist)                                        # 停滞 → LLM 第 1 次
    hist.append({"round": 3, "value": 0.69, "accepted": False, "lr": 0.001, "epochs": 8})
    h.decide(hist)                                        # LLM 第 2 次（1 次未采纳）
    hist.append({"round": 4, "value": 0.69, "accepted": False, "lr": 0.0011, "epochs": 8})
    plan = h.decide(hist)                                 # 连续 2 次未采纳 → 熔断
    assert c.calls == 2 and "LLM 熔断" in plan.hypothesis and h.last_degraded is None
    # 第 5 轮有新上线 → 下一次 decide 时重置熔断计数（此处走快通路，不停滞）
    hist.append({"round": 5, "value": 0.75, "accepted": True, "lr": 0.003, "epochs": 5})
    h.decide(hist)
    assert h.llm_misses == 0 and c.calls == 2
    # 再次制造停滞，LLM 应恢复调用
    hist.append({"round": 6, "value": 0.74, "accepted": False, "lr": 0.003, "epochs": 5})
    hist.append({"round": 7, "value": 0.74, "accepted": False, "lr": 0.0015, "epochs": 5})
    h.decide(hist)
    assert c.calls == 3


def test_hybrid_breaker_can_be_disabled():
    c = CountingClient('{"epochs": 8, "learning_rate": 0.001, "hypothesis": "换轴"}')
    h = HybridDecider(RulePolicy(base_epochs=5, base_lr=0.003), decider(c), llm_patience=0)
    hist = [{"round": 1, "value": 0.70, "accepted": False, "lr": 0.003, "epochs": 5},
            {"round": 2, "value": 0.70, "accepted": False, "lr": 0.0015, "epochs": 5}]
    for k in range(3, 7):
        h.decide(hist)
        hist.append({"round": k, "value": 0.69, "accepted": False, "lr": 0.001 * k, "epochs": 8})
    assert c.calls == 4
