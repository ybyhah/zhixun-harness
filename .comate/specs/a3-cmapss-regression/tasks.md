# A3 C-MAPSS 回归适配任务计划

> 对应 `.comate/specs/a3-cmapss-regression/doc.md`。
> 原则：扩展而非改写，A1 分类路径必须始终保持可用（每步末尾跑一次 A1 测试回归验证）。

- [x] Task 1: 回归指标与分数方向统一
    - 1.1: 在 `monitor/performance.py` 新增 `rmse` 与 C-MAPSS 非对称评分 `asym_score`
    - 1.2: 新增统一入口：回归任务对外分数取 `-RMSE`（越大越好），报告层转回正 RMSE
    - 1.3: 扩展 `evaluate` 支持 `task_type="regression"`，返回 RMSE、非对称分、统一 score
    - 1.4: `score_model` 对回归模型返回统一 score，空数据返回 NaN
    - 1.5: 分类路径的返回结构与字段保持不变
    - 1.6: 编写测试：RMSE/非对称分数值正确，方向统一（更好的预测 score 更大）

- [x] Task 2: C-MAPSS 数据适配器
    - 2.1: 新增 `datasets/cmapss.py`，解析 `train_FDxxx.txt`（26 列空格分隔）
    - 2.2: 计算训练集 RUL（max_cycle - cycle），按 `rul_cap` 分段线性截断
    - 2.3: 剔除全程常数的无效传感器，记录剔除列表
    - 2.4: 实现 `task_type="regression"`、`metric_name`、`class_set=None`
    - 2.5: 编写测试：FD001 轨迹数 100、列数、RUL 非负且被 cap 截断、无效传感器被剔除

- [x] Task 3: 批次流构造
    - 3.1: 实现方案 A（FD001→FD003→FD002→FD004 渐进），每子集内按 unit 分组切批
    - 3.2: 实现方案 B（单子集按 unit 号切批）作为简单对照
    - 3.3: 批次划分方案由配置选择（`stream.mode: progressive | single`）
    - 3.4: 复用现有 `BatchStream`，确认回归标签（连续值）在串行卡点下正常返回
    - 3.5: 编写测试：两种方案批次数正确、不泄漏未来数据、每批样本量合理

- [x] Task 4: 回归模型支持
    - 4.1: 给 `IndustrialModel` 增加 `task_type` 参数，回归用 HistGradientBoostingRegressor
    - 4.2: 回归分支下 `predict` 返回连续值，禁用 `predict_proba`/`class_bias`/`calibrate`
    - 4.3: 保留 scaler 快照机制与时间衰减权重（回归同样适用）
    - 4.4: 自适应迭代数与下采样对回归生效（下采样回归不分层，随机抽样）
    - 4.5: 编写测试：回归模型拟合、预测连续值、save/load 一致、分类路径不受影响

- [x] Task 5: 回归漂移判定
    - 5.1: `monitor/distribution.py` 增加目标值分布漂移（对 y 做 KS 检验）
    - 5.2: `monitor/report.py` 增加回归判定分支：用"目标分布漂移"替代"标签占比漂移"
    - 5.3: 性能下降判定改为基于统一 score（-RMSE 下降即性能下降）
    - 5.4: 保留 STABLE / COVARIATE_BENIGN / COVARIATE_HARMFUL / CONCEPT / UNTRUSTED 语义
    - 5.5: 编写测试：注入工况漂移后判为 COVARIATE_HARMFUL，目标漂移被检出

- [x] Task 6: 回归动作表
    - 6.1: `decide/rules.py` 增加回归动作阶梯：窗口重训 / 全量重训 / 特征子集（去掉校准、调阈值）
    - 6.2: 回归的 concept 漂移同样用全量重训 + 时间衰减
    - 6.3: 小批次保护、动作升级逻辑复用
    - 6.4: 编写测试：回归各漂移类型触发预期动作，不产生分类专用动作

- [x] Task 7: 主闭环按任务类型分支
    - 7.1: `loop.py` 根据 `cfg.task.type` 构造回归或分类的模型、监控、决策
    - 7.2: 冷启动、门控、回放集、审计记录在回归下正常工作（回放集回归不分层）
    - 7.3: 门控三道检查在回归统一 score 下方向正确
    - 7.4: 编写集成测试：A3 流跑完全部批次、无人干预、产出 ≥3 条完整闭环记录

- [x] Task 8: A3 配置与实验脚本
    - 8.1: 新增 `configs/a3_cmapss.yaml`（任务类型、数据路径、rul_cap、批次方案、各阈值）
    - 8.2: `scripts/run_baseline.py` 与 `run_loop.py` 支持 `--config` 跑 A3（或新增 a3 脚本）
    - 8.3: 回归指标的 `metrics.summarize` 适配（最差阶段、恢复速度基于统一 score）
    - 8.4: 产出 `outputs/e0_baseline_a3.csv`、`e2_rule_loop_a3.csv`、对照曲线

- [x] Task 9: A3 实验与结论
    - 9.1: 跑 E0 静态基线，确认 FD001→FD004 漂移流上 RMSE 显著上升
    - 9.2: 跑 E2 规则闭环，对比综合 RMSE、最差阶段、恢复速度、训练次数
    - 9.3: 固定种子验证可复现
    - 9.4: 运行完整测试套件，确认 A1 全部测试仍通过
    - 9.5: 更新 `outputs/CONCLUSIONS.md`：加入 A3 结果与跨任务（分类/回归）共用闭环的结论
