# 给 SDK 提交补真实的「更新与否」决策

- [x] Task 1: 新增 `submission/decision.py`（numpy-only，零新依赖）
    - 1.1: 从 `src/driver/significance.py`/`official_metric.py` 裁出 numpy 版配对 bootstrap 与主指标方向判定
    - 1.2: 对外 `should_promote(parent_pred, cand_pred, y, task, p_threshold=0.95) -> (bool, info)`
    - 1.3: bootstrap/预测不可用时回退聚合比较；再不行默认保留父模型

- [x] Task 2: 在 `train()` 接入门控
    - 2.1: 续训后加载父/候选检查点，在验证投影上预测，调 `should_promote`
    - 2.2: 始终 `submit_candidate`；`selected_artifact_id` = 候选或 `rnd.parent.artifact_id`
    - 2.3: `reason`/`configuration` 写明比较结果（p_better、采纳/保留父模型）；每轮仍一次模型调用

- [x] Task 3: 测试补强
    - 3.1: `submission/decision.py` 单测：全对 vs 全错 → 显著更好；相同预测 → 不晋升
    - 3.2: `scripts/local_sdk_test.py` 新增「候选更差 → 保留父模型」用例，断言选中父 artifact_id

- [x] Task 4: 验证
    - 4.1: `python3 scripts/local_sdk_test.py` 晋升/保留两条路径都过；`describe` 正常
    - 4.2: `python3 -m pytest -q` 既有用例不受影响
