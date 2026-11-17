# 第二阶段 PyTRiO 大模型慢通路完成总结

## 结果：慢通路已接入，离线自动降级，不破坏无人干预

`src/driver/llm.py` 新增 `LLMClient` + `LLMDecider`（实现 `policy.Decider`，可直接替换 `RulePolicy`）。大模型只影响"下一轮续训超参"的离线决策，不在延迟关键路径；任何失败降级到规则快通路且绝不抛错。

- `run_official.py --decider llm` 在**无 PyTRiO 端点**环境下运行：全程降级到规则，结果与 `--decider rule` **完全一致**（A 0.714→0.726），证明慢通路不破坏无人干预（C1）。
- 审计每轮可见：LLM 假设（或 `[LLM降级:原因]` 前缀）、`llm_degraded` 标志、`cost.tokens`。
- 257 个测试全绿，含 `tests/test_llm.py` 11 项（prompt 含历史与基线、解析、越界裁剪、坏 JSON/超时/无端点均降级、永不抛错）。

## 改了什么

| 文件 | 改动 |
|---|---|
| `src/driver/llm.py` | 新增 `LLMClient`（OpenAI 兼容，urllib，无新依赖）+ `LLMDecider`（读历史→提 epochs/lr，内置规则降级） |
| `src/driver/loop.py` | 审计记录 LLM 假设/降级/token（小改） |
| `scripts/run_official.py` | 加 `--decider {rule,llm}`（默认 rule） |
| `tests/test_llm.py` | 新增 11 项离线 mock 测试 |

## 与比赛要求对齐

- 仅用平台千问（OpenAI 兼容），key/base_url/model 走环境变量，缺则不可用→降级。
- 大模型不在延迟关键路径；受控（轮数上限不变，只在续训约束内调超参）；可审计（假设/降级/token 全记录）。
- C4 离线评测：无端点即规则快通路，不影响跑完。

## 本轮范围与后续

- 本轮只做"超参假设"（读历史→提下一轮 epochs/lr），安全可测。
- 后续（独立 spec）：LLM 读写 `baselines/` 代码做模型层改动；策略层自改进（受控 RSI 第二层）。
