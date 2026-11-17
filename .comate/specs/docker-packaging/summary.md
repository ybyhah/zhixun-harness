# Docker 一键运行打包完成总结

## 结果：C5 / C11 两项硬门槛已满足

| 项 | 要求 | 实测 |
|---|---|---|
| 镜像大小 | ≤2GB | **402MB** |
| 不打包权重与数据 | C5 | 镜像内无 `base/`、`data/`、`outputs/`、`third_party/`、`backup/`（已逐项核验） |
| 一键运行 | C11 | `docker run` 一条命令跑完三包 + 自动硬约束自检 |
| 离线可跑 | C4 | `--network none` 下跑通 |
| 资源 | C3 | `--cpus 4 --memory 8g` 下跑通，峰值内存 249MB |
| 可复现 | C11 | 容器与宿主机同种子逐轮**逐位一致** |

## 关键设计取舍

**1. 镜像只装官方评分路径的依赖**
实测官方包 `baselines/requirements.txt` 只要 `numpy==1.26.4`，我们的 `src/driver/` +
`scripts/run_official.py` 的第三方依赖也只有 numpy（其余全标准库）。
`pandas/sklearn/scipy/matplotlib/frouros` 只被自研对照闭环使用，**不进镜像**。
用临时虚拟环境只装 numpy 验证过官方路径能完整跑通。镜像因此只有 402MB。

**2. `base/` 不打进镜像，运行时挂载**
官方说明数据包"上传至指定平台固定目录下"，由平台提供；同时避开任何"打包权重"的争议。
缺挂载时**明确报错 + 给挂载示例 + 退出码 2**，不静默跑空。平台目录名不同可用 `BASE_DIR` 覆盖。

**3. 自检脚本改为动态发现官方包**
原先写死了 `cpu-a-guide-20261110-v1` 这类带版本日期的目录名——平台目录名可能不同。
改为扫描 `BASE_DIR` 下的 `cpu-*` 并读 `baseline-info.json` 的 `dataset_key`，容器内外都适用。

## 新增/修改

| 文件 | 内容 |
|---|---|
| `Dockerfile` | 新增，`python:3.11-slim` + 最小依赖 + 固定 `PYTHONHASHSEED`/`BASELINE_CPU_THREADS` |
| `run.sh` | 新增，一键入口（校验挂载、按 `TASKS/ROUNDS/GATE/DECIDER/SWANLAB` 跑、末尾自检、透传参数） |
| `.dockerignore` | 新增，排除 base/data/outputs/backup/third_party/swanlog 及缓存与大文件 |
| `requirements-docker.txt` | 新增，镜像内最小依赖 |
| `requirements.txt` | 顶部注明"镜像不用此文件；这些依赖仅自研对照需要" |
| `scripts/verify_contest_conditions.py` | 改为动态发现官方包，支持 `BASE_DIR` |
| `RUNBOOK.md` | 新增，完整复现文档（构建/运行/环境变量/实测结果/自检/产出清单） |

310 个测试全绿；官方三包产出已还原。

## 容器内实测（离线 + 4 核 8G）

| 包 | 轮数 | 官方基线 | 我们最优 | 优于基线 | 闭环 | 耗时 | 峰值内存 |
|---|---|---|---|---|---|---|---|
| A | 3 | 0.7140 | 0.7258 | 是 | 3 | 2.9s | 67MB |
| B | 3 | 681.16 | 666.88 | 是 | 3 | 4.9s | 73MB |
| C | 3 | 52.87 | 52.23 | 是 | 3 | 34.2s | 249MB |

容器内自检八项全过。

## 过程中的一个外部阻碍

Docker CLI 在位（29.5.2）但守护进程没启动，构建报 socket 不存在。启动 Docker Desktop 后
20 秒内就绪，构建正常。提交前在新机器上复现时注意这一步。

## 提交物进度

| 提交物 | 状态 |
|---|---|
| 系统镜像（≤2GB、一键运行） | **完成**（402MB） |
| 源码与配置、复现文档 | **完成**（`RUNBOOK.md`） |
| 实验记录链接 | 已上传 SwanLab，**待设为公开**（C8，需账号操作） |
| 设计说明 | 待起草 |
| 技术报告（≤10 页） | 待起草 |
| 演示视频（≤5 分钟） | 待录制（可先写脚本） |
| 来源声明 | 待起草（素材在 `outputs/REFERENCES.md`） |
| 模型调用信息 | 待定（是否后训练 / 声明仅用官方基座） |
