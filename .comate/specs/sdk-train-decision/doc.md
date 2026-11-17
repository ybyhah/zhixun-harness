# 给 SDK 提交补真实的「更新与否」决策（sdk-train-decision）

## 要解决的问题
当前 `submission/contest_submission.py` 的 `train()` 是**退化决策**：每轮续训后无条件选中新候选。漂移下盲目每轮更新会过拟合当前批、拖累下一批（赛题「是否过拟合」扣分项）。本步把真正的「要不要上线」决策补进去：**候选不显著优于父模型时，保留父模型**。

## 关键设计依据（已核对官方代码）
- 官方 `baselines.train` 产出的 `training.json` 同时含 `validation_before`（父模型在本轮验证投影上的分）与 `validation_after`（候选的分），同口径同投影、可直接比较；不需要另造数据，也不违反「不自报分数」（平台仍独立测量，这只是本地决策依据）。
- 指标方向可靠地由 task 推导：classification → Macro-F1 越大越好；regression → RMSE 越小越好。不依赖 `rnd.task` 里不确定的字段。
- SDK 允许三种选择：选新候选 / 保留父模型 / 选回更早产物（`finish` 的 `selected_artifact_id` 可为 `rnd.parent.artifact_id`）。本步先做「候选 vs 父模型」两选一。
- `rnd.parent.path("checkpoint.npz")` 可拿到父检查点；候选检查点在 `/workspace/cand/checkpoint.npz`。两者都能本地加载做配对显著性检验。

## 技术方案（两层，默认用更稳的一层，可降级）
1. **聚合门控（必做，稳健）**：读 `training.json` 的 before/after，按方向判断候选是否「优于父模型且超过极小阈值 ε」。否则保留父模型。ε 防止追噪声。
2. **配对显著性门控（增强，默认开）**：把父/候选两个检查点都在验证投影上预测，做配对 bootstrap（搬 `src/driver/significance.py` 的实现到 `submission/decision.py`，零外部依赖、与官方指标口径一致）。`p_better ≥ 阈值`（默认 0.95）才选候选，否则保留父模型；bootstrap 不可用时回退第 1 层。
- 模型调用仍每轮一次（提超参 + 产出 hypothesis），**决策门控独立于模型**：即使模型降级，门控照常保护。
- 候选始终 `submit_candidate`（让平台测量留痕），但 `finish` 的 `selected_artifact_id` 由门控决定（候选或父模型）；`reason` 如实写明比较结果。

## 依赖与选型（已调研 OSS，结论：不引新依赖）
- 漂移检测类 OSS（river/frouros/nannyml）项目里已集成，但只在 `scripts/bench_external.py` 做本地对照，**不进提交镜像**（镜像按规定仅 numpy）。
- 重型持续学习框架（mammoth/DER++/AMR 等）面向可训练 NN + replay，和「固定官方 NumPy MLP、梯度只用训练批、提交 checkpoint」不兼容，不采用。
- 「有标签时该不该上线」是 champion/challenger + 配对显著性检验；**复用自家 numpy-only、口径对齐官方指标、已做 split-half 验证的 `src/driver/significance.py`**，搬成 `submission/decision.py`，零新依赖、离线可跑。

## 影响文件
- 新增 `submission/decision.py`：`paired_bootstrap` + 主指标方向判定（从 `src/driver/significance.py`/`official_metric.py` 裁出 numpy-only 版本），对外 `should_promote(parent_pred, cand_pred, y, task, p_threshold) -> (bool, info)`；bootstrap 不可用时回退聚合比较。
- 改 `submission/contest_submission.py` 的 `train()`：续训后加「候选 vs 父模型」门控，决定 `selected_artifact_id`。
- 改 `scripts/local_sdk_test.py`：加「候选更差 → 保留父模型」用例。

## 实现细节（train 决策流）
1. 续训产出候选 `cand/checkpoint.npz`，读 `training.json` 的 `validation_before/after`（兜底聚合比较）。
2. 加载父检查点与候选检查点，在**验证投影**上各自预测 → `submission.decision.should_promote(...)`：
   - 配对 bootstrap `p_better ≥ 0.95` 判候选显著更好；否则判保留父模型。
   - 预测/ bootstrap 不可用（异常）→ 回退：按方向比较 `validation_after` 是否优于 `validation_before` 超过 ε。
3. 始终 `submit_candidate`（平台留痕测量）；`finish` 的 `selected_artifact_id` = 候选或 `rnd.parent.artifact_id`，`reason` 写明「p_better=… 采纳/保留父模型」。
4. 模型调用照旧每轮一次（提超参 + hypothesis），门控独立于模型；两者都写进 `configuration`。

## 边界与异常
- 门控任何异常（加载失败/维度不符/bootstrap 失败）→ 回退聚合比较；再不行默认**保留父模型**（更保守、抗过拟合），绝不中断本轮。
- 首轮父模型即官方基线：候选不显著更好就保留基线，符合「不低于基线」。
- 仍每轮一次模型调用、每轮 `submit_candidate` + `finish`；`model_call_ids` 非空不变。
- 自评仅用于本地决策；对外不另报分数（平台测量为准）。

## 本地测试
- `scripts/local_sdk_test.py` 现有用例保持；新增：构造「候选更差」场景（如给退化超参或直接桩掉预测），断言 `finish` 选中 `rnd.parent.artifact_id`、reason 含「保留父模型」。
- 另加 `submission/decision.py` 的单测：全对 vs 全错预测 → 显著更好；相同预测 → 不晋升。

## 预期结果
- 提交物具备真实「更新与否」决策：候选不显著优于父模型时保留父模型，直接对应「稳定性/是否过拟合/最差阶段」。
- 零新依赖、镜像仍 numpy-only、离线可跑；`local_sdk_test.py` 两条路径（晋升/保留）都通过；`pytest` 不受影响。

