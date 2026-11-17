# 改训练损失任务计划

> 对应 `.comate/specs/custom-training-loss/doc.md`。
> 原则：不改 `base/` 里任何官方文件（子类 + 运行时注入）；只改训练损失，推理协议与 checkpoint 格式不变；
> α=1/权重全 1 必须与官方损失数值等价（可关自证）；默认路径行为不变；结论按预设阈值判定，无效如实记录。

- [x] Task 1: WeightedModel（子类重写 fit）
    - 1.1: 新增 `src/driver/weighted.py` 的 `WeightedModel(官方 Model)`，只重写 `fit`，其余继承
    - 1.2: 分类：类别加权交叉熵，`w_c = N/(K·n_c)` 归一化到均值 1，权重按全训练集频次预算（与 batch 组成无关）
    - 1.3: 回归：非对称平方损失，高估（err>0）乘 α；α=1 退化为官方损失
    - 1.4: 可选 Huber（δ 外线性），用于 B 的对照
    - 1.5: 保留官方梯度裁剪（norm 5.0）、Adam 更新、steps/epochs 计数与 validate 调用

- [x] Task 2: 运行时注入与入口
    - 2.1: `weighted.py` 生成 `patched_train.py` 启动器：`import baselines.train as T; T.Model = WeightedModel; T.main()`
    - 2.2: `trainer.py` 新增 `entry_override`：用指定脚本替代 `-m baselines.train`（与 `profile_override` 同风格）
    - 2.3: `loop.py` 透传 `entry_override`
    - 2.4: 新增 `scripts/run_loss_variant.py --package <dir> --loss {weighted_ce,asym,huber} --alpha A --rounds 10`
    - 2.5: 产物写 `outputs/loss/<profile>/<变体>/`，不动 `base/`

- [x] Task 3: NASA 非对称分
    - 3.1: `metric.py` 新增非对称分计算（官方 `train.py` 的 metrics 不含此项）
    - 3.2: C 的运行结果额外报告非对称分（越低越好）
    - 3.3: 测试：高估比同幅低估罚更重

- [x] Task 4: 自证与测试
    - 4.1: `tests/test_weighted.py`：α=1 时 fit 与官方 Model 逐步等价（权重/损失数值一致）
    - 4.2: 权重全 1 的加权交叉熵与官方交叉熵等价
    - 4.3: 非对称损失梯度方向正确（高估惩罚更大）
    - 4.4: 产出 checkpoint 能被**官方未修改的** `Model.load` 与 predict 路径加载
    - 4.5: 运行完整 `python3 -m pytest` 全绿

- [x] Task 5: 对照实验
    - 5.1: A：官方交叉熵（0.7264）vs 类别加权，10 轮
    - 5.2: C：官方（51.38）vs α ∈ {1.5, 2, 3}，10 轮，同时报 RMSE 与非对称分
    - 5.3: B：官方（666.88）vs α ∈ {1.5, 2} 与 Huber，10 轮
    - 5.4: 固定种子复跑确认可复现

- [x] Task 6: 判定与收尾
    - 6.1: 按阈值判定（A > 0.004；B/C RMSE 相对改善 > 1%；C 非对称分改善 > 5% 亦算有效）
    - 6.2: 更新 `outputs/CONCLUSIONS.md`：损失对照表与采纳结论
    - 6.3: 采纳则在提交材料声明"修改训练损失、不影响推理协议"；不采纳则如实记录原因
    - 6.4: 确认默认路径（`run_official.py`）结果与改动前一致
