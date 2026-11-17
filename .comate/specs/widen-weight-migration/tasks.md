# 函数保持式加宽 + 权重迁移任务计划

> 对应 `.comate/specs/widen-weight-migration/doc.md`。
> 原则：迁移必须**函数保持**（迁移后指标与原模型逐位一致，不一致即报错）；不改官方 `baselines/`；
> 不污染 `base/`；未加宽的默认路径行为完全不变；结论按"显著优于才采纳"判定，无效则如实记录。

- [x] Task 1: 迁移工具（函数保持 + 自证）
    - 1.1: 新增 `src/driver/migrate.py` 的 `widen_checkpoint(src, dst, new_hidden, seed, eps)`：扩 `w1`（新列 σ=eps 随机）、`b1`（新位 0）、`w2`（**新行置 0**，保证输出不变）
    - 1.2: Adam 一阶/二阶矩按槽位迁移，新槽置 0；`mean/scale/target_mean/target_scale/seed/steps/epochs` 原样继承
    - 1.3: 更新 `meta.hidden` 与 `meta.parameter_count`；迁移前校验参数量不超官方上限（<100000）
    - 1.4: 自证：迁移后在验证集上的主指标必须与原模型一致（容差 0），否则抛错
    - 1.5: `tests/test_migrate.py`：函数保持（logits 逐元素相等）、形状/元信息正确、Adam 新槽为 0、参数量超限报错、非加宽（new<=old）报错

- [x] Task 2: 驱动器支持自定义起点与 profile
    - 2.1: `trainer.py` 支持传入自定义 profile JSON 路径（官方 `train.py` 已支持 `--profile *.json`）
    - 2.2: `loop.py`/`package.py` 支持指定初始 checkpoint 与 profile 覆盖；门控基线仍用官方随包基线值
    - 2.3: 默认路径（`run_official.py` 不带加宽）行为与改动前完全一致
    - 2.4: 测试：自定义 profile 时命令拼装正确（mock 子进程）

- [x] Task 3: 加宽运行入口
    - 3.1: 新增 `scripts/widen_and_run.py --package <dir> --hidden 64 --rounds 10`：迁移 → 生成只改 `hidden` 的 profile JSON → 跑满 10 轮
    - 3.2: 迁移产物与轮次输出写到 `outputs/widen/<profile>/hidden-<H>/`，不动 `base/`
    - 3.3: B 自动带 label-provenance（复用现有逻辑）

- [x] Task 4: 公平对照实验（A/B/C × 原始/64/128）
    - 4.1: A 三组各 10 轮，记录最优指标、上线/训练次数、耗时、参数量
    - 4.2: B 三组各 10 轮
    - 4.3: C 三组各 10 轮（耗时较长，可后台跑）
    - 4.4: 固定种子各复跑一次确认可复现

- [x] Task 5: 判定与收尾
    - 5.1: 按阈值判定是否采纳（A 提升需 > 0.004；B/C 相对改善 > 1%）
    - 5.2: 更新 `outputs/CONCLUSIONS.md`：三组对照表 + 采纳结论 + 合规依据（函数保持自证）
    - 5.3: 保留未加宽结果作可回退方案，两套 checkpoint 都留在 `outputs/`
    - 5.4: 运行完整 `python3 -m pytest` 全绿
    - 5.5: 无论正负如实记录；若无效，说明"容量不是瓶颈"的判断依据
