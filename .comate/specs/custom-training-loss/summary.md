# 改训练损失完成总结

## 结论：三种损失全部更差，不采纳

实现方式**子类 + 运行时注入**：`WeightedModel` 只重写 `fit`，启动器里 `T.Model = WeightedModel` 后调官方
`train.main()`。不改 `base/` 任何文件；官方 train.py 的全部校验与报告、Adam、梯度裁剪、checkpoint 格式都复用，
官方未修改的 `predict.py` 能原样加载产物。

**可关自证（单测 atol=0 逐位一致）**：权重全 1 的加权交叉熵、α=1 的非对称损失与官方损失完全等价，
证明改动可关、无隐藏副作用。

## 实验结果

| 任务 | 官方损失 | 实验损失 | 结果 | 判定 |
|---|---|---|---|---|
| A Macro-F1↑ | **0.7264** | 类别加权 CE | 0.7244 | 更差 |
| B RMSE↓ | **666.88** | 非对称 α=1.5 / 2、Huber δ=1 | 678.09 / 676.05 / 667.32 | 全部更差 |
| C RMSE↓ | **51.38** | 非对称 α=1.5 / 2 / 3 | 52.37 / 未上线 / 未上线 | 全部更差 |

**C 的非对称损失最值得记录**：它把高估比例从 0.570 压到 0.500，但 **NASA 非对称分反而从 3.64e9 升到 7.78e9**，
RMSE 也退化。该分数由 `exp(d/10)` 主导、被最大的几个高估决定——压低高估的*比例*无助于压低高估的*幅度*。
α≥2 时 RMSE 退化到门控一次都不放行。

## 新增/修改

| 文件 | 内容 |
|---|---|
| `src/driver/weighted.py` | 新增：`class_weights`、`make_weighted_model`（加权 CE / 非对称 / Huber）、`write_launcher` |
| `src/driver/trainer.py` | 新增 `entry_override`（用指定脚本替代 `-m baselines.train`） |
| `src/driver/loop.py` | 透传 `entry_override` |
| `src/driver/metric.py` | 新增 `nasa_asym_score`（标注为 C-MAPSS 专用）、`over_estimate_rate` |
| `scripts/run_loss_variant.py` | 新增：按损失变体跑满 N 轮，回归任务附报高估比例、C 附报非对称分 |
| `tests/test_weighted.py` | 新增 11 项（等价自证、权重均值为 1、非对称方向、Huber 抗离群、官方可加载、启动器内容） |
| `outputs/CONCLUSIONS.md` | 损失对照表 + 性能探索总账 |

全部 290 个测试通过；默认路径（`run_official.py`）结果与改动前一致（A 0.7140→0.7264）。

## 过程中修掉的一个报告 bug

最初对 B 也算了 NASA 非对称分，得到 1e51 这种无意义数字——该分数是 C-MAPSS（循环）专用口径，
`exp(d/10)` 对 B 的分钟量级（可达上千）会溢出。已改为只对 profile C 计算，B 只报高估比例。

## 性能探索到此结束

超参、容量、损失三条路都做完了同等条件对照，全部无效。**最终采用官方原始架构 + 官方损失 +
规则策略驱动的 10 轮续训**（A 0.7264 / B 666.88 / C 51.38，均优于官方基线）。

剩余提分空间只在非性能项：专家分 65 与提交硬门槛（SwanLab 公开、Docker、技术报告、演示视频、来源声明）。
这四个方向的负结果总账本身就是"实验规范与分析"的强材料。
