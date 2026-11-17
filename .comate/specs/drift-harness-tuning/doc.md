# 第一阶段优化：恢复速度、资源效率与指标口径

> 承接 `.comate/specs/drift-harness-mvp/`（已完成）。
> 本阶段只做优化，不引入大模型，不扩展新任务。
> 目标是把第一阶段暴露的三个问题修掉，并把客观分四项指标变成可量化输出。

## 1. 为什么做这些

第一阶段跑完后有三个明确短板，都直接对应客观分：

| 问题 | 现象 | 影响的客观分项 |
|---|---|---|
| 单批耗时过长 | 总 62s，批次 6 全量重训 19.4s。评测是 4 核，可能翻倍 | 资源效率 5 分；超时还会触碰门槛 |
| 小窗口重训亏损 | 批次 2 更新后反而 -0.0524（窗口 3 但只有 2 批可用） | 漂移综合性能 15 分 |
| 漂移未恢复 | 批次 9 检测到漂移、触发重训，候选被门控拒绝后就不再尝试 | 最差阶段与恢复速度 10 分 |

另外"恢复速度""抗遗忘""资源效率"目前只是粗略打印，没有统一口径的量化输出，没法支撑对照实验的结论。

## 2. 四项优化

### 2.1 训练预算与自适应迭代数

**现状**：`max_iter` 固定 200，不论样本量多少都跑满。批次 6 全量重训时样本累积到近万条，耗时 19.4s。

**改法**：按训练样本量自适应设定迭代数，并给单次训练设样本上限（超出时分层下采样）。

```python
def adaptive_params(base: dict, n_samples: int, budget: TrainBudget) -> dict:
    p = dict(base)
    if n_samples <= budget.small_n:
        p["max_iter"] = budget.min_iter
    else:
        # 样本越多迭代越少，总计算量大致恒定
        scaled = int(budget.min_iter * budget.small_n / n_samples)
        p["max_iter"] = max(budget.floor_iter, min(base.get("max_iter", 200), scaled))
    return p
```

同时加 `max_train_samples`：超限时按类别分层抽样，保证稀有类不被挤掉。

**预期**：单批最大耗时降到 5s 内，总耗时降到 20s 内；性能损失要控制在可接受范围（通过对照验证）。

### 2.2 窗口最小样本量下限

**现状**：`WINDOW_RETRAIN` 固定取最近 k 个批次，早期批次少、样本不足时训练出的候选很弱。

**改法**：窗口按"凑够最小样本量"动态扩展，而不是固定批次数。

```python
def window_for_samples(pool, k_pref: int, min_samples: int) -> int:
    k = min(k_pref, pool.n_batches)
    while k < pool.n_batches and pool.window(k)[0].shape[0] < min_samples:
        k += 1
    return k
```

**预期**：批次 2 的更新不再是负收益。

### 2.3 单批次多候选竞争

**现状**：一次决策只产出一个候选，被门控拒绝就等下一批再试。批次 9 因此一直没恢复。

**改法**：决策器返回**一组**候选动作（按成本升序），执行器在护栏允许的训练次数内逐个尝试，门控择优上线。

- 决策层新增 `decide_many(report, history) -> list[Decision]`，规则决策器给出动作阶梯的前 N 项
- 主闭环在 `max_trainings_per_batch` 预算内依次训练并门控，取通过且分数最高者
- 全部被拒时按原逻辑计失败、累计到上限则回滚
- 所有尝试都记入 `experiments`，这正是"做实验"的证据

**预期**：恢复事件里的"未恢复"减少；训练次数会上升，但受护栏约束，且本地开发不计入资源消耗。

### 2.4 客观分指标口径统一

新增 `src/harness/metrics.py`，把四项指标算法固定下来，E0 到 E4 共用：

| 指标 | 定义 |
|---|---|
| 漂移综合性能 | 各批次指标的均值（跳过冷启动批次） |
| 最差阶段 | 滑动窗口（默认 2 批）均值的最小值，比单批最小值更稳 |
| 恢复速度 | 跌破"此前最佳 - tol"后恢复所需批次数；未恢复记为惩罚值 |
| 稳定性与抗遗忘 | 每次上线前后在回放集上的分数差，取最差一次 |
| 资源效率 | 单位性能增益消耗的训练次数与秒数 |

口径写死在代码里并有测试，避免不同实验用不同算法导致结论不可比。

## 3. 影响文件

```
修改 src/harness/models/industrial.py    自适应迭代数 + 训练样本上限
修改 src/harness/decide/actions.py       TrainingData 增加按样本量取窗口
修改 src/harness/decide/rules.py         新增 decide_many
修改 src/harness/loop.py                 多候选竞争；接入预算与指标
新增 src/harness/metrics.py              客观分四项指标统一实现
修改 configs/a1_gas.yaml                 新增 budget 段与 window 下限
修改 scripts/run_loop.py                 输出统一指标；支持 --variant 对照
新增 tests/test_metrics.py               指标口径测试
修改 tests/test_models.py                自适应参数与下采样测试
修改 tests/test_decide.py                多候选与窗口扩展测试
修改 tests/test_loop.py                  多候选竞争的集成测试
```

## 4. 边界条件

| 情况 | 处理 |
|---|---|
| 自适应后迭代数过小 | 设 `floor_iter` 下限，不低于该值 |
| 下采样导致类别缺失 | 分层抽样，每类至少保留 1 条 |
| 多候选全部被拒 | 与原逻辑一致：计失败、到上限回滚 |
| 训练预算在多候选中耗尽 | 停止尝试，已尝试的仍记入 experiments |
| 恢复事件未恢复 | 记为惩罚值（默认总批次数），而不是丢弃，否则会美化结果 |
| 回放集为空 | 抗遗忘指标返回 NaN 并标注，不参与汇总 |

## 5. 预期结果

**量化目标**（相对第一阶段 E2 的 0.6760 / 0.4392 / 62s）：

1. 综合性能不低于 0.676，目标 0.70 以上
2. 最差阶段指标改善
3. 总耗时降到 25s 内，单批最大 5s 内
4. 恢复事件中的"未恢复"减少
5. 批次 2 的更新不再是负收益

**验收**：`python scripts/run_loop.py` 产出统一口径的指标表，与第一阶段结果并列对比，并说明每项优化各贡献了多少。做不到的要如实写出来。
