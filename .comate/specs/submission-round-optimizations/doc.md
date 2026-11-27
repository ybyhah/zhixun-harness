# 提交物单轮优化：把已验证的 Hybrid 决策移植进 SDK 提交物

## 先对照历史：哪些已证实、哪些已证伪

| 历史 spec | 结论 | 对本次的影响 |
|---|---|---|
| `official-multicandidate` | 每轮 3 候选：A 0.7258 vs 单候选 0.7264，B 持平，C 更差；epochs 15 比 5 更差（过拟合） | **不再做多候选**（上一版 doc 的 S3 撤回） |
| `custom-training-loss` / `widen-weight-migration` | 改损失、加宽都无收益 | 不再碰模型侧 |
| `significance-gate` | 配对 bootstrap 拦掉 p=0.606 的噪声峰，采纳为默认 | 提交物已有，保留 |
| `decider-comparison-eval` v1–v2 | 旧提示词的纯 27B：B 部署增益 +2.43，规则 +14.28；27B 不知道默认配置，常提 20–50 epochs | **当前提交物就是这种"纯 27B + 无锚点提示"** |
| `decider-comparison-eval` v3–v5 | 提示词加默认配置锚点、当前最优、每轮 epochs 与拒绝原因、禁止重复；Hybrid（规则快通路，停滞或爬坡放缓才用 27B，连续 2 次未采纳熔断）。split-half 留出裁决：B、C 3/3 显著优于规则，A 不更差 | **这是已验证最好的决策器，但没移植进 SDK 提交物** |
| `比赛要求汇总.md` §5.2、§3.3、§15 | 每批重发全部历史数据；恢复速度占 10 分，快速通路不能依赖大模型；必须有同等条件对照 | 决定代理实验设计和对照组 |

## 现状问题（已实测）

| # | 问题 | 证据 |
|---|---|---|
| P1 | 决策器退回到已被证明较差的"纯 27B、无锚点"形态，首轮就由 27B 定超参 | `contest_submission.py:47-77`；实测 27B 首轮提议 epochs=10、lr=0.001，偏离官方默认 5 epochs |
| P2 | 27B 读到的历史为空 | 代码取 `validation_after`/`metrics` 键，SDK 未定义记录结构，官方替身记录只有 `id`、`round_index` |
| P3 | 未关闭思考 | 同一提示：默认 7s、推理 733 字；`enable_thinking=False` 2s、无推理。官方系统提示最长 8000 字，推理可能耗尽 512 token，导致空回复和静默回退 |

已核对无问题：回归输出已截断非负（`baselines/model.py:159`）；三包单轮续训 1–8 秒、门控 0.1 秒。

## 方案

### S1 关闭思考（修 P3）

```python
try:
    receipt = rnd.model(msgs, max_tokens=n, temperature=0, enable_thinking=False)
except HarnessError:          # 宿主不接受该选项时，不带它再调一次
    receipt = rnd.model(msgs, max_tokens=n, temperature=0)
```

`enable_thinking` 在 SDK 白名单内（`round.py:15-19`）。

### S2 用 /state 保存轨迹（修 P2，也是 S3 的前提）

SDK 保证 `/state` 跨轮传递。`state/trail.json` 结构：

```json
{"rounds": [{"round": 1, "path": "rule", "epochs": 5, "lr": null, "value": 0.7166,
             "accepted": true, "p_better": 0.986, "reason": "", "parent": "<artifact_id>"}],
 "llm_misses": 0, "last_effective_lr": null, "baseline_value": 0.7140}
```

- 开头 `rnd.state_workspace(work/"state")` 读取；不存在或损坏就当空轨迹。
- `finish` 前（含续训失败、NOOP 分支）追加本轮并 `rnd.save_state(state)`；保存失败不影响本轮。
- `baseline_value` 首轮取父模型在验证集上的指标（`training.json` 的 `validation_before`）。

PLACEHOLDER_S3
