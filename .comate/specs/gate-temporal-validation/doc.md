# 门控时序验证：缓解门控分偏乐观

> 承接 `outputs/COMPARISON.md` 第 3 节：门控分系统性高于下一批真实表现（A1 +0.38、A2 +0.52）。
> 目标：给门控一个**时序隔离**的验证口径，让候选的门控分更接近它在下一批上的真实表现，并用 A/B + bench 量化是否改善上线决策。

## 1. 问题定位

现在门控验证集来自 `TrainingData.split_last`（`src/harness/decide/actions.py:104`）：从**最近一个批次**按类别分层切 30% 作验证，其余 70% 连同更早批次进训练池。

- 候选确实没在这 30% 上训练（out-of-sample），但这 30% 与训练数据**同分布、同时刻**。
- 门控分反映的是"同分布泛化"，而真正要预测的是"下一批（已漂移）的表现"。两者的差就是漂移间隙，`bench_external.py --dim estimate` 实测 A1 +0.38、A2 +0.52。
- 后果：门控分普遍 0.96~1.0，压在高位，`min_gain` 与 `not_below_baseline` 两道检查在被压缩的高分区里比较，差异淹没在噪声里。

## 2. 方案：可选的时序验证口径

给门控验证集加一个**时序隔离**选项，用配置项 `gate.val_mode` 选择，默认保持现状以不影响已有结论。

| `val_mode` | 训练池 | 验证集 | 说明 |
|---|---|---|---|
| `in_batch`（默认，现状） | 最近批次的 70% + 更早全部 | 最近批次的 30% 分层切片 | 同分布，偏乐观 |
| `holdout_recent`（新增） | **不含最近批次**的全部历史 | **整个最近批次** | 候选完全没见过验证集，且验证集是手上最新、最可能已漂移的数据，是对"下一批"最好的现成代理 |

`holdout_recent` 的逻辑：候选在"截至上一批"的窗口上训练，在"最新一批"上被门控打分。这正是时序模型选择里的 walk-forward 验证——用最近一段未见数据估计泛化，比同批切片更接近真实间隙。

**代价与取舍**：`holdout_recent` 下被提升的候选，其训练窗口截止到上一批，没学到最新批次。下一轮窗口会自然纳入该批次，所以只是**推迟一批**纳入最新数据，不是永久丢弃。这一取舍是否划算，用 A/B 实测，不预设结论。本轮先不做"提升后在全窗口重训"（那会每次上线多一次训练），留作下一步按数据决定。

不偷看未来：验证集仍是**已经到手并提交过**的最新批次，不涉及任何未来数据，串行卡点不变。

## 3. 影响文件

| 文件 | 改动 |
|---|---|
| `src/harness/decide/actions.py` | `TrainingData.split_last` 增加 `val_mode` 参数，新增 `holdout_recent` 分支：训练池为 `X[:-1]`，验证集为整个 `X[-1]` |
| `src/harness/loop.py` | `_after_submit` 调 `split_last` 时传入 `self.cfg.gate.get("val_mode", "in_batch")` |
| `configs/a1_gas.yaml`、`a2_xjtu.yaml`、`a3_cmapss.yaml` | `gate` 段加 `val_mode: in_batch`（显式默认，便于 A/B） |
| `scripts/run_loop.py` | `VARIANTS` 增加 `holdout-gate`（`gate.val_mode=holdout_recent`），便于一键 A/B |
| `tests/test_decide.py` 或 `test_gate.py` | 补 `split_last` 两种模式的单元测试 |

`holdout_recent` 下若只有 1 个批次（冷启动后第一次门控），没有"更早批次"可训练：此时回退到 `in_batch`，并在验证集口径上标注，避免训练池为空。

## 4. 验证方法

1. **门控偏乐观对比**：给 `bench_external.py --dim estimate` 增加一个 `holdout_recent` 口径的高估量，和现有 `in_batch` 并列，直接看高估幅度是否下降。
2. **端到端 A/B**：`run_loop.py --sweep` 或 `--variant holdout-gate`，在 A1/A2/A3 上对比两种 `val_mode` 的综合性能、最差阶段、恢复速度、上线次数、训练次数。
3. 固定种子可复现；A1/A3 的默认（`in_batch`）结果必须与改动前完全一致。

## 5. 预期结果（待实测验证，不预设）

- `holdout_recent` 的门控高估量应明显小于 `in_batch`。
- 端到端性能可能有两种走向，都如实记录：
  - 若综合性能持平或上升：时序验证让门控更准，采纳为默认或新增推荐。
  - 若因"推迟一批纳入最新数据"而下降：说明该取舍不划算，保留 `in_batch` 默认，把结论写进报告，并把"提升后全窗口重训"列为下一步。

## 6. 边界

| 情况 | 处理 |
|---|---|
| `holdout_recent` 但只有 1 批 | 回退 `in_batch` |
| 最新批次极小（如 A2 的 2_4 仅 42 条） | 验证集就是这 42 条，Macro-F1 已按本批出现类别算，可接受 |
| 回归任务（A3） | 同样适用，验证集换成整批，分数方向不变 |
| 未知 `val_mode` 取值 | 抛 `ValueError`，不静默回退 |
