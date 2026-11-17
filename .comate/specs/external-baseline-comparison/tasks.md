# 外部开源基线对比评测任务计划

> 对应 `.comate/specs/external-baseline-comparison/doc.md`。
> 原则：不改 harness `src/`，只新增对比脚本与报告。安装库、clone 仓库需联网，执行时请求升级权限。

- [x] Task 1: 脚手架与依赖
    - 1.1: 新增 `scripts/bench_external.py` 骨架：`--dim {drift,online,estimate,all}`、`--config`，复用 `build_adapter` 读批次
    - 1.2: 外部库按维度惰性 import，缺失时打印"未安装，跳过"而不崩
    - 1.3: 安装 river、nannyml（联网），更新 `requirements.txt` 的"仅本地对比"注释段
    - 1.4: 新增 `third_party/` 并加入忽略（不进镜像）

- [x] Task 2: 漂移检测对比（1.1）
    - 2.1: 从现有闭环复用静态基线逐批预测，产出三任务的 0/1 误差流与特征流
    - 2.2: 接 river 的 ADWIN、KSWIN、Page-Hinkley 与 frouros 的 DDM/HDDM，记录各自告警批次
    - 2.3: 与我们 monitor 的 drift 判定批次对齐，输出一致/漏报/误报对比表
    - 2.4: 写入 `COMPARISON.md` 的漂移检测小节

- [x] Task 3: 在线学习基线（1.2）
    - 3.1: 封装 River `HoeffdingAdaptiveTreeClassifier`、`ARFClassifier`，逐批 predict→评分→learn，严格不偷看未来
    - 3.2: 回归任务接 River 回归器同法（A3）
    - 3.3: 用现有 `metrics` 同口径算综合性能、最差阶段，与 E0、E2 并列
    - 3.4: 三任务输出 E0 / E2 / River-online 对比表，写入报告

- [x] Task 4: 无标签性能估计（1.3）
    - 4.1: A1/A2 用 `IndustrialModel.predict_proba` 接 NannyML CBPE，在下一批上估 Macro-F1
    - 4.2: 三方对比：门控候选分 vs NannyML 估计 vs 下一批真实分
    - 4.3: 量化门控偏乐观幅度；采用"当前批内留出集 vs 下一批真实"的简化估计（见 summary 偏差说明）
    - 4.4: 写入报告

- [x] Task 5: 参照仓库只读对比（1.4）
    - 5.1: `git clone --depth 1` aideml 到 `third_party/`，阅读其历史利用与实验择优机制
    - 5.2: clone 一个 XJTU-SY 轴承特征/诊断仓库，对比特征集合与标签方案 vs 我们的方案
    - 5.3: 在报告里写参照要点与可借鉴项（不改代码）

- [x] Task 6: 汇总与轻量测试
    - 6.1: 整理 `COMPARISON.md`：四维结论 + 下一轮优化清单（本轮不实施）
    - 6.2: 新增 `tests/test_bench_external.py`：对脚本内纯函数（批次对齐、告警汇总）做单元测试
    - 6.3: 运行 `python3 -m pytest` 确认全绿、harness 行为不变
