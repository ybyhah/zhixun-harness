# 正式 SDK 闭环与漂移证据修复设计

## 目标

在 `contest.harness.workspace.v2` 正式提交路径中解决两个验收问题：平台轮次能够稳定形成可核验的完整闭环证据；漂移判定不依赖正式镜像中不存在的 `frouros`/ADWIN。

## 非目标

- 不把旧 `src/harness` 流式模拟器打入正式镜像；
- 不给正式镜像增加 `frouros`、River、SciPy 等依赖；
- 不使用本地验证指标替代平台 `submitted["summary"]`；
- 不改变 P1 的 A/B/C 训练策略和平台选型规则。

## 正式闭环证据

每次 `rnd.finish()` 的 `configuration.loop_evidence` 写入：

- `history_ids`：实际由 `rnd.history()` 返回并引用的记录；
- `history_read`：是否完成历史读取，首轮空历史仍为真；
- `model_call_ids`：本轮真实 `rnd.model()` 调用号；
- `hypothesis_recorded`：是否记录有效假设；
- `experiment_status`：`candidate_measured` 或 `train_failed`；
- `decision_source`：`platform_summary` 或 `failure_fallback`；
- `complete`：上述读历史、提假设、做实验、做决策均完成时为真。

成功训练轮必须在候选获得平台摘要后才将 `complete` 置真。训练失败仍属于已执行实验并依据失败结果保留父模型，因此记录为完整失败闭环。辅助函数只统计结构化 `loop_evidence.complete=true`，不再借用旧本地 `DecisionRecord.experiments` 的定义。

## 正式漂移判定

候选提交并获得平台摘要后，使用候选与当前父模型的同名主指标计算方向统一的相对增益：

- 越大越好：`(candidate - parent) / max(abs(parent), 1e-12)`；
- 越小越好：`(parent - candidate) / max(abs(parent), 1e-12)`。

相对增益低于 `-0.02` 判为 `HARMFUL`，否则分别记录为 `IMPROVED` 或 `STABLE`。指标缺失时状态为 `UNKNOWN`，检测器仍标记为启用但不伪造报警。C 包只要 NASA 非对称分相对父模型上升，也判为 `HARMFUL`，原因记为 `nasa_worsened`。

结果写入 `configuration.platform_drift`：

- `detector="platform_metric_delta"`；
- `enabled=true`；
- `drift_detected`；
- `status`、`reason`；
- 主指标、候选值、父值、相对增益和阈值。

该信号是平台客观性能漂移，不声称替代输入分布 PSI。它直接服务正式轮次的性能恢复和稳定性证据。

## Qwen 历史输入

`_history_prompt()` 在每条平台历史后附加上一轮 `platform_drift` 的 `status`、`reason` 和相对增益。Qwen 因而能在下一轮根据真实平台退化调整 epochs/lr；已有 B 停滞换轴继续作为确定性护栏。

## 旧本地测试对齐

- 旧 `src/harness` 的“至少三次闭环”测试改为本地模拟器自身的审计完整性检查，不再承担正式 SDK 赛题验收；正式三闭环证据由 `tests/test_submission_contract.py` 覆盖。
- 旧 ADWIN 测试在可选 `frouros` 不存在时跳过，因为生产代码本来就声明降级为空检测器；正式漂移能力由无依赖的平台指标检测测试覆盖。

## 验收标准

1. 三条正式历史记录均带完整 `loop_evidence` 时计数为 3。
2. 首轮空历史仍能记录 `history_read=true`。
3. 成功候选记录 `candidate_measured/platform_summary`；训练失败记录 `train_failed/failure_fallback`。
4. A/B/C 主指标相对退化超过 2% 时检测到漂移。
5. C 的 NASA 恶化能独立触发漂移。
6. 指标缺失时返回 `UNKNOWN`，不报警、不崩溃。
7. Qwen 提示包含最近平台漂移结论。
8. 正式镜像不新增依赖，SDK 测试、契约测试、Docker 构建与 describe 通过。

