# 门控 probe + 全窗口重训任务计划

> 对应 `.comate/specs/gate-probe-refit/doc.md`。
> 原则：默认 `in_batch` 行为与改动前完全一致；`holdout_refit` 用 A/B 量化后再决定去留。重训复用现有训练路径，受同一护栏约束。每步末尾跑 A1/A3 回归验证。

- [x] Task 1: split_last 支持 holdout_refit 切分
    - 1.1: `TrainingData.split_last` 的 `val_mode` 接受 `holdout_refit`，切分行为同 `holdout_recent`（probe 池 = `X[:-1]`，验证集 = 整个 `X[-1]`）
    - 1.2: 单批时回退 in_batch；未知值仍抛 `ValueError`
    - 1.3: `tests/test_decide.py` 补 holdout_refit 切分等价 holdout_recent 的测试

- [x] Task 2: _compete 支持选中后全窗口重训
    - 2.1: `_compete` 新增可选参数 `refit_pool`，默认 None 时行为完全不变
    - 2.2: promote 分支选出最优动作后，若 `refit_pool` 非空，用 `_try_action(idx, decision, refit_pool)` 重训部署模型
    - 2.3: 重训成功则 stage 重训模型，失败（单类/超时/异常）回退 probe 候选上线
    - 2.4: 重训计入 `used` 训练次数，受 `max_total_trainings`、`train_timeout_s` 护栏约束
    - 2.5: 全部被拒/无候选分支保持不变

- [x] Task 3: 闭环接入与变体
    - 3.1: `loop.py` 的 `_after_submit`：`val_mode=="holdout_refit"` 且 `self.pool.n_batches>=2` 时向 `_compete` 传 `refit_pool=self.pool`
    - 3.2: `scripts/run_loop.py` 的 `VARIANTS` 增加 `holdout-refit`
    - 3.3: 跑 A1/A3 默认配置，确认逐批分数与改动前完全一致

- [x] Task 4: 端到端 A/B
    - 4.1: A1/A2/A3 对比 `full` / `holdout-gate` / `holdout-refit` 的综合性能、最差阶段、恢复、上线/训练次数、耗时
    - 4.2: 固定种子各跑两次确认可复现
    - 4.3: 运行完整 `python3 -m pytest` 确认全绿、默认行为不变

- [x] Task 5: 结论与去留决策
    - 5.1: 汇总三者 A/B，判断 holdout_refit 是否 ≥ in_batch 的综合性能，且保持门控诚实
    - 5.2: 核对多出的训练次数与耗时，对照赛题资源效率项确认代价可接受
    - 5.3: 按结果决定是否采纳为推荐默认；更新 `outputs/COMPARISON.md` 第 3 节
    - 5.4: 如实记录收益与代价，收益不显著也写明原因
