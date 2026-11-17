# 门控加入严重退化召回检查任务计划

> 对应 `.comate/specs/gate-critical-recall/doc.md`。
> 原则：A1/A3 不配置该项，行为必须与改动前完全一致；仅 A2 启用，用前后对比量化 Severe 召回改善与综合性能代价。每步末尾跑回归验证。

- [x] Task 1: Gate 增加第四道检查
    - 1.1: `Gate.__init__` 增加 `critical_class`、`critical_recall_tol` 参数（默认 None，不启用）
    - 1.2: `GateResult` 增加 `candidate_critical_recall`、`online_critical_recall` 字段并纳入 `to_dict`
    - 1.3: Gate 内新增私有方法：模型在 (Xr,yr) 上 predict 后调 `evaluate(..., critical_class=...)` 取关键类召回
    - 1.4: `evaluate` 末尾加第四道检查：候选关键类召回不得低于在线 `critical_recall_tol`；候选召回 NaN（验证集缺该类）或在线召回 NaN 时跳过并在 checks 标注
    - 1.5: 仅当 `critical_class` 与 `critical_recall_tol` 都给定时启用

- [x] Task 2: 闭环与配置接入
    - 2.1: `loop.py` 构造 `Gate` 时传 `critical_class=self.critical_class`、`critical_recall_tol=cfg.gate.get("critical_recall_tol")`
    - 2.2: `configs/a2_xjtu.yaml` 的 `gate` 段加 `critical_recall_tol: 0.05`
    - 2.3: 跑 A1/A3，确认逐批分数与改动前完全一致（未配置项不触发）

- [x] Task 3: 单元测试
    - 3.1: `tests/test_gate.py` 补：关键类召回退化超容忍 → 拒绝
    - 3.2: 小幅退化在容忍内 → 通过
    - 3.3: 验证集缺关键类 / 无在线模型 → 跳过通过
    - 3.4: 未配置 `critical_recall_tol` → 第四道不触发，结果与旧逻辑一致
    - 3.5: 运行完整 `python3 -m pytest` 全绿

- [x] Task 4: A2 效果评估
    - 4.1: 跑 `run_loop.py --config configs/a2_xjtu.yaml`，对比加检查前后 E2 Severe 平均召回与综合 Macro-F1
    - 4.2: 固定种子两次确认可复现
    - 4.3: 若综合性能受损过多，试 `--set gate.critical_recall_tol=0.1` 找平衡点，记录召回 vs 综合性能权衡
    - 4.4: 查上线/训练次数变化，确认未因新检查异常增减

- [x] Task 5: 结论与去留
    - 5.1: 判断是否采纳（Severe 召回上升且综合性能可接受）
    - 5.2: 更新 `outputs/CONCLUSIONS.md` A2 小节：补严重退化召回门控的前后对比与权衡
    - 5.3: 如实记录；若两者强冲突则呈现权衡曲线而非硬压单值
