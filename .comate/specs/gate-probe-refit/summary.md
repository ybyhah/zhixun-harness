# 门控 probe + 全窗口重训完成总结

## 结论：又一个负结果，in_batch 仍是默认

实现了 `holdout_refit`：门控用 probe（训练到上一批）在最新批上诚实打分，选中动作后在全窗口重训部署模型再上线。它把上一轮 `holdout_recent` 丢掉的适应性基本补回来了，但**三个任务仍都不及 in_batch 默认**，且更费训练。

### 三种口径端到端 A/B（固定种子，两次一致）

| 任务 | in_batch（默认） | holdout_recent | holdout_refit | 训练次数 in→refit |
|---|---|---|---|---|
| A1 | **0.6079** | 0.5573 | 0.5967 | 9 → 13 |
| A2 | **0.5861** | 0.4393 | 0.5214 | 13 → 16 |
| A3 | **-22.07** | -24.36 | -24.70 | 7 → 12 |

### 为什么没涨：门控是相对比较，绝对乐观是良性的

门控三道检查都是"候选 vs 在线 vs 基线在**同一验证集**上"的相对比较。验证集绝对分偏高会同时作用于三方，相对排序不受影响。所以改善绝对校准（holdout）没用，反而因为 probe 更弱更严、拦掉有益更新、引入方差，把性能和训练开销都做差了。连续两轮（holdout_recent、holdout_refit）指向同一结论。

## 改了什么

| 文件 | 改动 |
|---|---|
| `src/harness/decide/actions.py` | `split_last` 的 `val_mode` 增加 `holdout_refit`（切分同 holdout_recent） |
| `src/harness/loop.py` | `_compete` 新增 `refit_pool` 参数：选中动作后在全窗口重训部署模型，失败回退 probe，受同一护栏约束；`_after_submit` 按 `val_mode` 传入 |
| `scripts/run_loop.py` | `VARIANTS` 增加 `holdout-refit` |
| `tests/test_decide.py` | 补 holdout_refit 切分等价测试 |
| `outputs/COMPARISON.md` | 第 3.2 节补实测结论 |

默认 `in_batch` 行为与改动前逐批分数完全一致（A1/A3 校验）。全部 220 个测试通过。无新增依赖，未碰镜像/离线边界。

## 与比赛要求的对齐

属模型层自迭代（受控 RSI 第一层）的工程探索，没有跑偏。两轮实验用对照证伪了"门控校准是瓶颈"这一假设，正好是"实验规范与分析"（主观分 15）的强材料：不是所有看起来更优的改动都真更优。`holdout_recent`/`holdout_refit` 作为变体保留，供答辩演示验证过程。

## 下一步建议（真正可能涨分的方向，已不在门控校准上）

门控校准这条线到此证明是死胡同，建议转向：
1. **门控加入 Severe 召回检查**（承接 A2 结论，直接针对严重退化漏报，对应客观分"稳定性"）。
2. **monitor 接入 ADWIN 快速触发**（frouros 现成），降低漏报，对应"恢复速度"10 分。
3. **第二阶段大模型慢通路**（按 aideml 的实验树 + 双角色评审 + 原子假设），对应方法设计 30 分与"策略层自改进"。
