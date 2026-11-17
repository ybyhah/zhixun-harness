# 门控加入严重退化召回检查

> 承接 A2 结论：规则闭环的 Severe 召回（0.699）反而**低于**静态基线（0.742）——闭环在严重退化识别上出现回归。
> 目标：给门控加一道"关键类召回不得明显退化"的检查，直接对准赛题 A2 的"严重退化漏报"与客观分"稳定性与抗遗忘"。

## 0. 与比赛要求的对齐

| 赛题要求 | 对齐 |
|---|---|
| A2 指标：Macro-F1 + **严重退化漏报** | 本检查直接约束 Severe 类召回，漏报率 = 1 − 召回 |
| 客观分"稳定性与抗遗忘"5 分 | 防止上线模型为提升整体 Macro-F1 而牺牲关键类 |
| "不更新也可以是正确决策" | 该检查只会**拦掉**损害关键类的候选，不会制造额外更新 |
| 可审计 | 门控记录新增关键类召回前后值，写入审计 |

纯第一层模型自迭代的门控增强，不引入任何新依赖，不碰镜像/离线边界，不跑偏。

## 1. 问题定位

`outputs/CONCLUSIONS.md` A2 第 4 节：E2 的 Severe 召回损失集中在突发失效轴承（批 5=2_1、批 10=3_1）。原因是当时在线模型从渐进退化轴承重训而来，对突发失效不敏感；而门控只看 Macro-F1，一个牺牲 Severe 召回但整体 Macro-F1 更高的候选也会被放行。

## 2. 方案：可选的第四道门控检查

在现有三道检查（优于在线、不低于基线、回放集不遗忘）之后，加第四道：

**关键类召回不退化**：候选在门控验证集上的关键类（A2=Severe，类别 2）召回，不得比在线模型低超过 `critical_recall_tol`。

- 仅当 `task.critical_class` 已配置且 `gate.critical_recall_tol` 已设置时启用；A1/A3 不配置，行为完全不变。
- 关键类在验证集中**未出现**时跳过该检查（无法度量，默认通过），避免因小批次缺类误伤。
- 在线模型召回为 NaN（无在线模型或无关键类样本）时跳过。
- 与前三道一样：任一不过即拒绝候选，不改变"全部被拒则计失败、达上限回滚"的既有逻辑。

## 3. 影响文件

| 文件 | 改动 |
|---|---|
| `src/harness/gate/evaluator.py` | `Gate.__init__` 增加 `critical_class`、`critical_recall_tol`；`evaluate` 末尾加第四道检查；`GateResult` 增加 `candidate_critical_recall`、`online_critical_recall` 字段与 `to_dict` |
| `src/harness/loop.py` | 构造 `Gate` 时传入 `critical_class=self.critical_class`、`critical_recall_tol=cfg.gate.get("critical_recall_tol")` |
| `configs/a2_xjtu.yaml` | `gate` 段加 `critical_recall_tol: 0.05` |
| `tests/test_gate.py` | 补：关键类召回退化超限被拒、小幅退化在容忍内通过、验证集缺关键类时跳过、未配置时不触发 |

主闭环、指标、审计口径其余不变。关键类召回复用 `monitor.performance.evaluate(..., critical_class=...)`（已实现）。

## 4. 实现要点

- Gate 内新增私有方法：对给定模型在 (Xr,yr) 上 `predict` 后调 `evaluate(..., critical_class=self.critical_class)` 取 `critical_recall`。候选与在线各算一次。
- 候选关键类召回为 NaN（验证集无该类）时，该检查判为通过并在 `checks` 里标注跳过。
- 检查顺序放在最后：前三道是整体性能与遗忘，第四道是关键类专项，语义上递进。

## 5. 验证方法

1. **A2 专项**：`run_loop.py --config configs/a2_xjtu.yaml` 对比加检查前后的 E2 Severe 平均召回（目标：从 0.699 提升，尽量不低于 E0 的 0.742），同时看综合 Macro-F1 是否明显受损。
2. 若综合性能下降过多，调 `critical_recall_tol`（放宽到 0.1）再看平衡点。
3. A1/A3 默认不配置该项，逐批结果与改动前完全一致。
4. `python3 -m pytest` 全绿。

## 6. 预期与去留（按数据）

- 期望：E2 Severe 召回上升，综合 Macro-F1 小幅波动可接受（关键类漏报对应独立的 5 分，值得小幅让渡整体）。
- 若 Severe 召回上升且综合性能基本持平：采纳，`critical_recall_tol` 写入 A2 默认。
- 若为保 Severe 召回导致综合性能大跌：说明两者强冲突，记录权衡曲线（不同 tol 的召回 vs 综合性能），由报告呈现，不硬压一个值。

## 7. 边界

| 情况 | 处理 |
|---|---|
| 验证集无关键类样本 | 跳过检查（通过） |
| 无在线模型（冷启动后首次） | 跳过（无可比对象） |
| 候选关键类召回 = 在线（持平） | 通过（未超容忍） |
| `critical_recall_tol` 未设 | 不启用第四道检查 |
| 回归任务（A3） | 无关键类概念，不配置，天然不触发 |
