# 完成总结：给 SDK 提交补真实的「更新与否」决策

## 做了什么
把提交物从「每轮盲目上线新候选」改成**有依据的上线决策**：候选不显著优于父模型时保留父模型。零新依赖、镜像仍 numpy-only、离线可跑。

- 新增 `submission/decision.py`（numpy-only）：配对 bootstrap + 官方口径指标（分类 Macro-F1 在 expected∪argmax 上宏平均、回归 -RMSE），`should_promote(parent_pred, cand_pred, y, task, p_threshold=0.95) -> (promote, info)`，退化输入返回 `None` 交上层回退。
- 改 `submission/contest_submission.py` 的 `train()`：续训后加载父/候选检查点在验证投影上预测 → `should_promote` 判定；bootstrap 不可用回退 `validation_before/after` 聚合比较；再不行默认**保留父模型**（保守抗过拟合）。始终 `submit_candidate` 留痕，`finish` 的 `selected_artifact_id` = 候选或 `rnd.parent.artifact_id`，`reason`/`configuration` 写明 `p_better` 与判定。每轮仍一次模型调用。
- 改 `scripts/local_sdk_test.py`：新增门控两条路径与真实门控、决策单测。

## 为什么不引 OSS（已调研）
- river/frouros/nannyml 已在 `bench_external.py` 做本地漂移检测对照，按规定不进镜像。
- 重型持续学习框架（mammoth/DER++/AMR）面向可训练 NN+replay，与固定官方 NumPy MLP 不兼容。
- 「有标签时该不该上线」= champion/challenger + 配对显著性检验，复用自家已验证的 numpy 实现最合适，不拖依赖进离线镜像。

## 验证（全通过）
```
[describe] ok
[decision/unit] ok: 全对>全错 显著；相同不晋升
[train/promote] ok: 选中候选
[train/keep] ok: 保留父模型
[decision/real] ok: promote=True p_better=0.986（真实父/候选检查点）
[predict] ok: 3 行
```
`pytest -q` 308 条全绿（未改 `src/`）。

## 边界
- 门控任何异常 → 回退聚合比较 → 再不行默认保留父模型，绝不中断本轮。
- 自评仅用于本地上线决策；平台独立测量为准，不对外另报分数。
- 端到端平台往返（`/previous`、`/model` 挂载）仍需镜像/平台试运行；本地用 FakeRound + 真实检查点验证了逻辑与真实门控。

## 残留与后续
- 决策基于「当前公开验证投影」；它与隐藏的下一批仍可能有分布差，p_threshold=0.95 已偏保守以抑制过拟合。
- 可选增强：支持选回更早产物（`restore_artifact`）、按 `rnd.round_index` 派生 bootstrap 种子。暂未做。
