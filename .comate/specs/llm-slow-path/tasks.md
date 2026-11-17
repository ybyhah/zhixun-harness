# 第二阶段 PyTRiO 大模型慢通路任务计划

> 对应 `.comate/specs/llm-slow-path/doc.md`。
> 原则：大模型不在延迟关键路径；任何失败降级到规则快通路且不抛错；不改官方 `baselines/`；默认 `--decider rule`，llm 可选。全程离线用 mock 验证。

- [x] Task 1: LLM 客户端（OpenAI 兼容，离线安全）
    - 1.1: 新增 `src/driver/llm.py` 的 `LLMClient`：读 `PYTRIO_BASE_URL`/`PYTRIO_MODEL`/`PYTRIO_API_KEY`，缺任一即 `available=False`
    - 1.2: 一次 chat.completions 调用，短超时（~20s），不重试风暴；优先用已装 `openai`，否则标准库 `urllib` POST（不新增依赖）
    - 1.3: 返回文本 + token 数；网络/非 200/超时抛内部异常由上层捕获
    - 1.4: 单测：无环境变量→available=False；mock 响应→返回文本与 token

- [x] Task 2: LLMDecider（实现 Decider，内置降级）
    - 2.1: `decide(history)` 构造紧凑上下文：profile/task/指标方向/官方基线 + 最近 N 轮 (round,value,accepted,lr)，提示"不重复失败设置，提一个原子改动"
    - 2.2: 要求返回 JSON `{epochs,learning_rate,hypothesis}`；解析 + 校验 epochs∈[1,50]、lr∈(0,0.1]，越界裁剪
    - 2.3: 任何失败（不可用/超时/坏 JSON/缺字段/越界无法裁剪）→ `fallback`(RulePolicy)，hypothesis 前缀 `[LLM降级:原因]`
    - 2.4: 记录 `last_tokens`/`last_degraded`/`last_raw`（截断）
    - 2.5: `decide` 永不抛错

- [x] Task 3: 闭环与入口接入
    - 3.1: `loop.py`：审计 `hypothesis` 用 LLM 假设，`cost.tokens` 填本轮 token，新增是否降级标注
    - 3.2: `run_official.py` 加 `--decider {rule,llm}`（默认 rule）；llm 时构造 `LLMDecider(fallback=RulePolicy(...))`
    - 3.3: 不改官方 `baselines/`；不改默认行为（rule）

- [x] Task 4: 测试（离线 mock）
    - 4.1: `tests/test_llm.py`：prompt 含历史与官方基线、合法 JSON 解析、越界裁剪、坏 JSON/无端点→降级规则、token 记录
    - 4.2: `decide` 在各失败路径均返回合法 RoundPlan 且不抛错
    - 4.3: 运行完整 `python3 -m pytest` 全绿

- [x] Task 5: 端到端与收尾
    - 5.1: `run_official.py --package base/cpu-a-guide-... --decider llm` 在无 PyTRiO 环境下运行，确认全程降级、结果与 `--decider rule` 一致（不破坏无人干预）
    - 5.2: 审计中可见每轮 LLM 假设/降级/token 字段
    - 5.3: 更新 `outputs/CONCLUSIONS.md`：慢通路接入说明 + "有端点才调用、否则规则"的离线策略
    - 5.4: 标注后续（LLM 改 baselines 代码、策略层自改进）入下一轮
