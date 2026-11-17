# 第二阶段：PyTRiO 大模型慢通路（续训决策）

> 承接 `harness-official-baseline`：规则策略已能驱动官方续训并超基线。本轮接入**大模型慢通路**，
> 让 PyTRiO Qwen 读实验历史、提出下一轮续训的超参假设，决策更有依据；对应"两层自进化"的策略层与方法设计分。

## 0. 与比赛要求对齐

| 要求 | 落实 |
|---|---|
| 仅用平台千问（27B 推荐），OpenAI 兼容 | `LLMDecider` 走 OpenAI 兼容客户端，base_url/model/key 来自环境变量；默认指向 PyTRiO |
| 大模型**不在延迟关键路径** | 大模型只影响"下一轮续训超参"的离线决策；预测/服务与训练本身不依赖它；不可用即退规则快通路 |
| 无人干预、不中断 | LLM 任何失败（无端点/网络/超时/坏 JSON/越界）都**降级到规则策略**并记录，绝不抛错 |
| 受控 RSI、不无限迭代 | 轮数上限不变；LLM 只在既有续训约束内调 epochs/lr，不能改架构/特征schema |
| 可审计 | 审计记录每轮的 LLM 假设、原始回复摘要、是否降级、token 数 |
| C4 离线评测 | 评测环境若无 PyTRiO 入口则自动走规则；有入口时才调用 |

不引入重框架（不装 AutoGen 等）；借 AIDE/OpenDrSai 的**压缩历史→原子假设→结构化回复**模式，用最小离线实现。

## 1. 范围（本轮只做"超参假设"，不做改代码）

LLM 慢通路第一步聚焦**读历史→提出下一轮续训超参**（epochs / learning_rate / 是否训）。
官方允许大模型"读写代码、用工具"，但自动改 `baselines/` 代码风险大、验证成本高，列为后续独立 spec。本轮把安全、高价值的超参决策做扎实并可测。

## 2. 设计 `src/driver/llm.py`

`LLMDecider` 实现既有 `policy.Decider` 协议（`decide(history) -> RoundPlan`），可直接替换 `RulePolicy`：

```
decide(history):
  1. 构造紧凑上下文：任务(profile/task/指标方向/官方基线)、最近 N 轮 (round,value,accepted,lr)
     —— 仿 AIDE 的 Memory 摘要，提示"不要重复已失败的设置，提一个可实验的原子改动"
  2. 调 OpenAI 兼容 chat.completions，要求返回 JSON：
     {"epochs":int, "learning_rate":float, "hypothesis":str}
  3. 解析 + 校验：epochs∈[1,50]，lr∈(0,0.1]，越界则裁剪；非法 JSON/缺字段 → 降级
  4. 返回 RoundPlan(epochs, learning_rate, hypothesis=LLM 理由, history_read)
  失败任一步 → self.fallback(RulePolicy).decide(history)，hypothesis 前缀标注"[LLM降级:原因]"
```

- 客户端：优先用已安装的 `openai`（若有）或标准库 `urllib` POST（避免新依赖）；base_url/model/key 读 `PYTRIO_BASE_URL`/`PYTRIO_MODEL`/`PYTRIO_API_KEY`，缺任一即视为不可用→降级。
- 轻量封装 `LLMClient`，超时短（如 20s），只做一次调用，不重试风暴。
- 记录 `last_tokens`、`last_degraded`、`last_raw`（截断）供审计与 SwanLab。

## 3. 闭环与入口改动

- `loop.py`：`cost.tokens` 填 LLM 本轮 token；审计 `hypothesis` 用 LLM 假设；新增字段标注是否降级。`loop` 已通过 `self.policy.decide` 调用，天然支持传入 `LLMDecider`，主逻辑几乎不改。
- `run_official.py`：加 `--decider {rule,llm}`（默认 rule）；选 llm 时构造 `LLMDecider(fallback=RulePolicy(...))`。
- 不改官方 `baselines/`。

## 4. 影响文件

| 文件 | 改动 |
|---|---|
| `src/driver/llm.py` | 新增 `LLMClient` + `LLMDecider`（实现 Decider，内置规则降级） |
| `src/driver/loop.py` | 审计记录 LLM 假设/降级/token（小改） |
| `scripts/run_official.py` | 加 `--decider` 开关 |
| `tests/test_llm.py` | 新增：prompt 含历史与基线、解析合法 JSON、越界裁剪、坏 JSON/无端点→降级到规则、token 记录 |

## 5. 验证（全程离线，用 mock，不依赖真实 PyTRiO）

1. 单测：mock 客户端返回合法/非法/超时响应，验证解析、裁剪、降级路径，`decide` 永不抛错。
2. 端到端：`run_official.py --decider llm` 在**无 PyTRiO 环境**下运行，应自动全程降级到规则，结果与 `--decider rule` 一致（证明慢通路不破坏无人干预）。
3. 若本地配置了 OpenAI 兼容端点（可选，不在 CI 强跑），手动验证一次真实调用并记录 token。
4. `python3 -m pytest` 全绿。

## 6. 边界与风险

| 情况 | 处理 |
|---|---|
| 无 PyTRiO 端点/密钥 | 视为不可用，全程规则快通路 |
| 网络超时/非 200/坏 JSON | 降级规则，记录原因与原始回复摘要 |
| LLM 返回越界超参 | 裁剪到合法区间；仍越界则降级 |
| LLM 建议改架构/特征 | 本轮不支持（只接受 epochs/lr），忽略并降级该字段 |
| token 成本 | 每轮一次调用、短超时；记录 token，对照资源效率项 |

## 7. 预期产出

1. `LLMDecider` 可插拔替换规则策略，离线自动降级，决策与降级全程可审计。
2. `--decider llm` 在无端点时与规则结果一致；有端点时用 LLM 假设驱动续训。
3. 为后续"LLM 改代码 / 策略层自改进"留好接口与记录骨架。
