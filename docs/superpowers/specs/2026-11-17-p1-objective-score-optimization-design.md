# P1 客观评分优化设计

## 目标

在不改变 `contest.harness.workspace.v2` 契约、不让本地验证分替代平台分的前提下，按顺序完成五项 P1 优化：保存最佳 epoch、改善 A 前两轮、B 停滞后换策略、C 抑制寿命高估、支持恢复平台历史最佳产物。

## 范围与边界

只修改正式提交链路：

- `submission/baselines/train.py`：逐 epoch 选择本轮最佳检查点；
- `submission/baselines/model.py`：C 包回归使用轻量非对称平方损失；
- `submission/contest_submission.py`：A/B 轮次策略与历史最佳产物选择；
- `tests/test_submission_contract.py` 和新增的 baseline 优化测试；
- 必要的运行/作品说明。

平台 `submitted["summary"]["metrics"]` 仍是上线与历史恢复的唯一计量来源。本地 `training.json` 只负责挑选同一次训练内部的 epoch，并作为诊断记录，不与平台候选或父模型跨产物比较。

## 1. 保存最佳 epoch

`train_run()` 仍按请求执行全部 epoch。每个 epoch 后计算公开验证指标：A 以 `macro_f1` 越大越好，B/C 以 `rmse` 越小越好。训练开始前的父模型只作为 `validation_before`，不会被包装成新候选；本轮至少训练一个 epoch，并在本轮各 epoch 中保存验证主指标最好的模型状态作为 `checkpoint.npz`。

报告新增 `selected_epoch`、`best_validation` 和 `trained_final_epochs`，同时 `validation_after` 明确表示最终写入检查点的验证指标。`history.jsonl` 仍保留所有实际执行的 epoch，便于审计。

## 2. A 包前两轮策略

A 的第 1、2 轮采用已有平台报告中表现最有希望的短搜索起点 `epochs=10, learning_rate=0.01`。Qwen 调用、假设和平台计量仍完整保留；编排层在记录中注明固定的 early-round profile guardrail。最佳 epoch 保存避免第 10 个 epoch 比中间 epoch 更差时丢失本轮最好状态。

该规则只作用于 A 的前两轮，后续完全回到 Qwen 提议与重复配置规避逻辑。

## 3. B 包停滞换策略

如果 B 的上一轮平台判定不是 `candidate_better` 或 `first_round_valid_candidate`，本轮从以下最小确定性序列中选择尚未在当前父模型上尝试的配置：

1. `(5, 0.001)`；
2. `(3, 0.0005)`；
3. `(8, 0.0003)`。

这使一次无改善后立即换轴，同时避免引入随机搜索或额外候选训练。没有可识别的平台停滞记录时仍使用 Qwen 提议。

## 4. C 包寿命高估惩罚与安全偏置

C 的回归训练在归一化目标空间使用非对称平方损失：预测高于真实寿命时权重为 `1.5`，低估时权重为 `1.0`。权重写入 checkpoint 元数据和训练报告；A/B 保持原损失不变。

本地 C 验证报告增加标准 NASA 非对称分用于诊断。平台选择仍执行现有安全门控：RMSE 改善但 NASA 分恶化时保留父模型。不会在 `predict()` 中直接减常数，以免对所有工况产生不可校准偏移。

## 5. 恢复平台历史最佳产物

从 `rnd.history()` 中解析平台指标，只考虑当前 start frame 的 `rnd.artifacts` 明确列出的可选产物。按任务主指标方向找到历史最佳记录，并与当前父模型、当前候选比较：

- 历史产物严格优于当前父模型时才有资格恢复；
- C 包历史产物若 NASA 分比当前父模型差，不恢复；
- 当前候选仍先按现有平台门控判断；随后在候选、父模型和合法历史最佳之间选择平台主指标最优且通过 C 安全约束的产物；
- `reason` 和 `configuration` 记录最终来源与历史记录 ID。

SDK 已允许 `finish()` 选择 start frame 中列出的更早产物，因此不复制历史文件，也不增加上传成本。

## 错误处理

- 最佳 epoch 指标缺失或非有限：该 epoch 不参与最佳选择；若所有 epoch 均非法，训练失败并由现有逻辑保留父模型。
- 平台历史记录结构不完整、产物不在 `rnd.artifacts` 或主指标非法：忽略该历史记录。
- Qwen 回复失效：沿用现有安全回退，再应用 A/B 的确定性 guardrail。
- C checkpoint 没有新权重字段：按 `1.0` 兼容加载；只有 C 续训时写入 `1.5`。

## 验收标准

1. 测试证明训练输出 checkpoint 来自本轮验证最优 epoch，而不是最后 epoch。
2. 测试证明 A 前两轮使用 `10/0.01`，第三轮不强制覆盖 Qwen。
3. 测试证明 B 一轮无改善后立即换到未尝试配置。
4. 测试证明 C 高估样本的梯度/损失权重大于低估样本，A/B 不变。
5. 测试证明历史最佳只从平台指标和平台允许的产物中选择。
6. 测试证明 C 不会恢复 NASA 更差的历史产物。
7. 正式 SDK 契约测试、本地训练替身、Docker 构建与 `describe` 通过。

