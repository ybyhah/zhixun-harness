# 第一阶段任务计划：工业模型自进化系统 MVP 骨架

> 对应 `doc.md`。目标是不依赖大模型跑通 A1 流式自适应闭环，产出 E0 与 E2 对照结果。

- [x] Task 1: 搭建项目骨架与依赖
    - 1.1: 创建 `requirements.txt`，补装 lightgbm、frouros、swanlab
    - 1.2: 建立 `src/harness/` 包目录与各子模块空文件
    - 1.3: 实现 `config.py`：YAML 加载与必填项校验
    - 1.4: 创建 `configs/a1_gas.yaml`，含路径、阈值、门控参数、随机种子
    - 1.5: 建立 `data/raw/`、`outputs/`、`tests/` 目录

- [x] Task 2: A1 数据接入与任务适配层
    - 2.1: 定义 `datasets/base.py` 的 TaskAdapter 协议：load_batches、metric、task_type、class_set
    - 2.2: 实现 `scripts/download_data.py`：下载 UCI Gas Sensor Array Drift，校验 10 个批次文件
    - 2.3: 实现 `datasets/gas_sensor.py`：解析 libsvm 风格 .dat，输出 Batch(X, y, index)
    - 2.4: 统一缺失值填充与标准化，scaler 只用已见批次拟合
    - 2.5: 固定全局类别集合，处理批次缺类问题
    - 2.6: 编写测试：验证 10 批次形状、128 维特征、各批类别分布与公开资料一致

- [x] Task 3: 串行批次流与漂移模拟器
    - 3.1: 实现 `stream/batch_stream.py`：next_batch 只返回 X，submit 后才返回 y
    - 3.2: 加入状态校验，未提交就取下一批、重复提交均抛异常
    - 3.3: 支持 submit 接受"下线判断"而非预测
    - 3.4: 实现 `stream/simulator.py`：协变量漂移（渐变偏移、缩放、噪声）
    - 3.5: 实现标签漂移（类别重采样）与概念漂移（局部标签重映射）
    - 3.6: 支持突变与渐变两种速率，参数由配置控制
    - 3.7: 编写测试：验证卡点约束生效、注入漂移后分布指标确实变化

- [x] Task 4: 工业模型与版本库
    - 4.1: 实现 `models/industrial.py`：LightGBM 封装，fit/predict/predict_proba
    - 4.2: 加入概率校准与阈值调整接口
    - 4.3: 支持样本时间衰减权重与滑动窗口训练
    - 4.4: 实现 save/load，模型体积控制在 MB 级
    - 4.5: 实现 `models/registry.py`：online/candidate/history 三态管理
    - 4.6: 实现 promote、discard、rollback 到历史最优
    - 4.7: 编写测试：上线与回滚后模型行为与预期一致

- [x] Task 5: 监测层
    - 5.1: 实现 `monitor/performance.py`：Accuracy、Macro-F1，处理缺类情况
    - 5.2: 实现 `monitor/distribution.py` 的 PSI 计算（分箱、边界与零频处理）
    - 5.3: 加入逐特征 KS 检验（scipy）
    - 5.4: 加入 PCA 重构误差作为多变量漂移信号
    - 5.5: 实现 `monitor/error_stream.py`：Frouros ADWIN/DDM 跑在误差序列上
    - 5.6: 实现 `monitor/report.py`：DriftReport 数据类
    - 5.7: 实现漂移类型判定（对应 doc.md 第 3.3 节判定表）
    - 5.8: 编写测试：对人工注入的三类漂移，判定结果正确

- [x] Task 6: 决策层（规则版）
    - 6.1: 定义 `decide/actions.py` 的 Action 枚举与 Decision 数据类
    - 6.2: 实现各动作执行器：不动、调阈值、重校准、增量、窗口重训、全量重训、特征子集
    - 6.3: 实现下线/拒判动作（标注待确认规则）
    - 6.4: 定义 Decider 协议，为第二阶段大模型决策器预留
    - 6.5: 实现 `decide/rules.py`：按判定表选动作，按成本从低到高尝试
    - 6.6: 加入小批次保护：样本量低于阈值只累积不更新
    - 6.7: 编写测试：各漂移类型触发预期动作

- [x] Task 7: 门控层
    - 7.1: 实现 `gate/replay.py`：分层抽样维护历史工况回放集，含容量上限与淘汰
    - 7.2: 实现 `gate/evaluator.py` 第一道检查：优于在线模型且超过 min_gain
    - 7.3: 实现第二道检查：不低于静态基线
    - 7.4: 实现第三道检查：回放集上退化不超过容忍度
    - 7.5: 实现连续失败计数与自动回滚触发
    - 7.6: 处理回放集为空的首批情况
    - 7.7: 编写测试：构造退化候选，验证被正确拒绝并回滚

- [x] Task 8: 记录层
    - 8.1: 实现 `record/schema.py` 的 DecisionRecord，字段留扩展位
    - 8.2: 实现 `record/audit.py`：JSONL 追加写与读取历史
    - 8.3: 实现 outcome 事后回填机制
    - 8.4: 实现 `record/swan.py`：SwanLab 封装，记录对象为被训练的工业模型
    - 8.5: 加入 SwanLab 不可用时的本地离线降级
    - 8.6: 编写测试：记录可写可读，字段完整

- [x] Task 9: 主闭环编排
    - 9.1: 实现 `loop.py`：按 doc.md 第 2.2 节七步串起全流程
    - 9.2: 保证第一步预测输出永不被后续异常阻塞
    - 9.3: 加入冷启动分支：首批无在线模型时的处理
    - 9.4: 加入安全护栏：单批训练次数上限、超时、总迭代上限
    - 9.5: 加入异常兜底：任一环节失败则降级并记录，不中断闭环
    - 9.6: 编写集成测试：人为注入训练失败与超时，验证仍能跑完 10 批

- [x] Task 10: E0 静态基线实验
    - 10.1: 实现 `scripts/run_baseline.py`：批次 1 训练，批次 2 到 10 只预测不更新
    - 10.2: 输出 `outputs/e0_baseline.csv` 逐批指标
    - 10.3: 记录单批次耗时，评估 4 核环境的时间余量
    - 10.4: 确认漂移导致的性能下降幅度并写入结论

- [x] Task 11: E2 规则闭环实验与对照
    - 11.1: 实现 `scripts/run_loop.py`：完整闭环跑 A1
    - 11.2: 输出 `outputs/e2_rule_loop.csv` 与 `outputs/audit.jsonl`
    - 11.3: 绘制 `outputs/curve_e0_e2.png` 对照曲线
    - 11.4: 统计综合性能、最差批次、恢复批次数、回放集退化、训练次数
    - 11.5: 验证产出至少 3 条含完整闭环字段的审计记录
    - 11.6: 固定随机种子，验证两次运行结果一致
    - 11.7: 撰写阶段结论：漂移有多严重、规则闭环救回多少、瓶颈在哪
