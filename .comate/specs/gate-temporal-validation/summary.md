# 门控时序验证完成总结

## 结论：负结果，保留 in_batch 默认

给门控加了时序验证口径 `holdout_recent`（候选训练到上一批为止，门控在整个最新批上打分），用 A/B + bench 两个角度量化。**门控更可信了，但端到端更差，因此不采纳为默认。**

### 门控偏乐观明显下降（bench，`--dim estimate`）

| 任务 | in_batch 平均高估 | holdout_recent 平均高估 |
|---|---|---|
| A1 | +0.3761 | +0.1867 |
| A2 | +0.5190 | -0.0040 |

### 但端到端性能全面下降（run_loop A/B）

| 任务 | in_batch 综合 | holdout_recent 综合 |
|---|---|---|
| A1 | 0.6079 | 0.5573 |
| A2 | 0.5861 | 0.4393 |
| A3 | -22.07 | -24.36 |

**机制**：holdout_recent 把最新批次整个排除出训练窗口来换取"验证集未被见过"，结果被提升的模型没学到最新漂移，适应性损失盖过了门控更准的收益。固定种子两次运行一致。

## 改了什么

| 文件 | 改动 |
|---|---|
| `src/harness/decide/actions.py` | `split_last` 增加 `val_mode` 参数与 `holdout_recent` 分支（单批回退 in_batch，未知值抛错） |
| `src/harness/loop.py` | 门控切分传入 `gate.val_mode` |
| `configs/*.yaml` | 三个配置 `gate` 段显式加 `val_mode: in_batch` |
| `scripts/run_loop.py` | `VARIANTS` 增加 `holdout-gate` 便于一键 A/B |
| `scripts/bench_external.py` | `run_estimate` 并列输出 in_batch / holdout_recent 两种口径的高估 |
| `tests/test_decide.py` | 新增 4 个 `split_last` 测试 |
| `outputs/COMPARISON.md` | 第 3.1 节补时序验证实测结论 |

默认行为（`in_batch`）与改动前逐批分数完全一致（A1/A3 已校验）。全部 219 个测试通过。

## 价值与下一步

- 价值：用数字证明"门控校准更好 ≠ 上线决策更好"。只优化门控而牺牲对最新漂移的适应会倒退。这条负结果可直接写进答辩的"实验规范与分析"。
- **已验证的正确方向（下一轮独立做）**：两步候选——先用"截至上一批"的窗口训 probe 做诚实门控打分，通过后再在**含最新批的全窗口**重训该动作才上线。同时拿到诚实门控与完整适应，代价是每次上线多一次训练（上线次数少，可接受）。预计这是真正能涨分的改法。
