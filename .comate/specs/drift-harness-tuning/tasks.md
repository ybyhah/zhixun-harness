# 优化阶段任务计划：恢复速度、资源效率与指标口径

> 对应 `.comate/specs/drift-harness-tuning/doc.md`。
> 每完成一项都要重跑 `scripts/run_loop.py`，记录该项单独的贡献，便于最后拆解收益。

- [x] Task 1: 建立优化前基准与指标统一实现
    - 1.1: 把当前 E2 结果另存为 `outputs/e2_before_tuning.csv`，作为优化前对照
    - 1.2: 新增 `src/harness/metrics.py`，实现漂移综合性能（跳过冷启动批次）
    - 1.3: 实现最差阶段（滑动窗口均值的最小值，窗口可配）
    - 1.4: 实现恢复速度（跌破"此前最佳 - tol"后的恢复批次数，未恢复记惩罚值）
    - 1.5: 实现稳定性与抗遗忘（上线前后回放集分数差，取最差一次；回放集空时返回 NaN）
    - 1.6: 实现资源效率（单位性能增益消耗的训练次数与秒数）
    - 1.7: 实现 `summarize()` 汇总入口，返回统一结构供各实验共用
    - 1.8: 新增 `tests/test_metrics.py`，覆盖各指标的边界（空序列、全 NaN、未恢复、单批次）

- [x] Task 2: 训练预算与自适应迭代数
    - 2.1: 在 `configs/a1_gas.yaml` 新增 `budget` 段：small_n、min_iter、floor_iter、max_train_samples
    - 2.2: 在 `models/industrial.py` 实现 `adaptive_params`：按样本量缩放 max_iter，受 floor_iter 约束
    - 2.3: 实现分层下采样 `subsample`：超过 max_train_samples 时按类别抽样，每类至少 1 条
    - 2.4: 在 `IndustrialModel.fit` 中接入自适应参数与下采样，并把实际用量记入 TrainStats
    - 2.5: 补充 `tests/test_models.py`：样本越多迭代越少、不低于 floor_iter、下采样后类别不丢失
    - 2.6: 重跑闭环，记录耗时与性能变化，确认单批最大耗时下降

- [x] Task 3: 窗口最小样本量下限
    - 3.1: 在 `decide/actions.py` 的 TrainingData 增加 `window_for_samples`，按凑够样本量扩展窗口
    - 3.2: 在配置新增 `loop.min_window_samples`
    - 3.3: 在执行器的窗口类动作中改用该方法确定窗口大小
    - 3.4: 补充 `tests/test_decide.py`：样本不足时窗口自动扩展、已足够时不扩展、不超过可用批次
    - 3.5: 重跑闭环，确认批次 2 的更新不再是负收益

- [x] Task 4: 单批次多候选竞争
    - 4.1: 在 `decide/actions.py` 定义 `decide_many` 的协议签名与返回结构
    - 4.2: 在 `decide/rules.py` 实现 `decide_many`：返回动作阶梯前 N 项（N 可配）
    - 4.3: 保留 `decide` 作为 `decide_many` 的首项，确保旧调用与第二阶段接口不受影响
    - 4.4: 在 `loop.py` 改为在训练预算内依次尝试候选，门控择优上线
    - 4.5: 记录每个候选的动作、参数、耗时、门控结果到 `experiments`
    - 4.6: 全部被拒时维持原失败计数与回滚逻辑
    - 4.7: 预算耗尽时停止尝试，已尝试的仍完整记录
    - 4.8: 补充 `tests/test_loop.py`：多候选择优、预算约束生效、全拒后回滚仍正常
    - 4.9: 重跑闭环，确认恢复事件中的"未恢复"减少

- [x] Task 5: 实验脚本与对照输出
    - 5.1: 在 `scripts/run_loop.py` 接入 `metrics.summarize()`，输出统一指标表
    - 5.2: 新增 `--variant` 参数，支持关闭单项优化以做消融对照
    - 5.3: 输出 `outputs/summary_tuning.csv`：优化前后与各消融组合的指标并列
    - 5.4: 绘制 `outputs/curve_tuning.png`：E0、优化前 E2、优化后 E2 三条曲线
    - 5.5: 验证固定种子下两次运行结果一致

- [x] Task 6: 消融实验与结论
    - 6.1: 依次关闭三项优化各跑一次，得到单项贡献
    - 6.2: 汇总各项对综合性能、最差阶段、恢复速度、耗时的影响
    - 6.3: 更新 `outputs/CONCLUSIONS.md`：加入优化阶段结果与消融拆解
    - 6.4: 如实记录未达预期的项目及原因分析
    - 6.5: 运行完整测试套件，确认全部通过
