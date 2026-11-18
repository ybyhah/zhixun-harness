# 运行说明（正式提交与本地复现）

> 评分对象是官方 `base/` 三个包里的 CPU NumPy MLP，从随包 `checkpoint.npz` **续训**。
> 本项目是驱动这一续训过程的 Harness。

## 1. 正式平台 SDK 镜像

正式提交使用 `contest.harness.workspace.v2`，只构建 `submission/Dockerfile`：

```bash
docker buildx build --platform linux/amd64 --provenance=false --sbom=false --load \
  -t zhixun-submit -f submission/Dockerfile .
```

固定入口由平台驱动：

```text
python3 /app/harness.py describe
python3 /app/harness.py train
python3 /app/harness.py predict
```

模型只通过 SDK 的 `rnd.model()` 调用，PyTRiO 密钥由平台持有。公开训练/验证投影、父产物和工作目录分别由平台挂载到 `/data`、`/previous` 和 `/workspace`，不使用下面的本地 `base/` 挂载命令。

正式链路会保存本轮验证最优 epoch；A 前两轮使用固定强起点，B 在一轮无改善后自动换轴，
C 使用寿命高估惩罚并同时检查平台 NASA 指标。最终产物只从当前候选、父模型和平台授权的
历史 artifact 中按平台 `summary.metrics` 选择。

每轮完成记录还包含：

- `configuration.loop_evidence`：历史读取、Qwen 调用、实验状态、决策来源及完整性；
- `configuration.complete_loops_total`：正式 SDK 完整闭环累计数；
- `configuration.platform_drift`：无外部依赖的平台主指标相对漂移，C 同时检查 NASA。

`platform_metric_delta` 是正式评分口径下的性能漂移检测，不是输入特征 PSI。旧本地模拟器的
ADWIN 依赖 `frouros`，未安装时按设计降级，不影响正式提交镜像。

本地只验证 `describe`：

```bash
docker run --rm zhixun-submit describe
```

> 根目录 `Dockerfile` 和 `run.sh` 是旧版本地实验镜像，不能作为正式平台提交物。

## 2. 本地实验镜像（非平台提交物）

| 项 | 值 |
|---|---|
| 镜像大小 | **402MB**（C5 要求 ≤2GB） |
| 基础镜像 | `python:3.11-slim` |
| 镜像内依赖 | 仅 `numpy==1.26.4` + `swanlab`（见 `requirements-docker.txt`） |
| 镜像**不含** | `base/`（官方包与工业模型权重）、`data/`、`outputs/`、`third_party/`、`backup/` |

镜像刻意不打包 `base/`：官方说明数据包"上传至指定平台固定目录下"，由平台提供；
同时避免任何"打包权重"的争议。运行时挂载即可。

### 构建本地实验镜像

```bash
docker build -t zhixun-harness .
```

## 3. 本地一键运行

```bash
mkdir -p outputs-docker
docker run --rm --network none --cpus 4 --memory 8g \
  -v "$PWD/base:/app/base:ro" \
  -v "$PWD/outputs-docker:/app/outputs" \
  zhixun-harness
```

默认行为：依次跑 A/B/C 各 10 轮续训，跑完自动做赛题硬约束自检。
`--network none` 用来证明**完全离线可跑**（C4）。

### 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `BASE_DIR` | `/app/base` | 官方包目录；平台目录名不同时用它覆盖 |
| `TASKS` | `a,b,c` | 要跑的任务，逗号分隔 |
| `ROUNDS` | `10` | 每包续训轮数 |
| `GATE` | `significance` | `plain` \| `significance`（配对 bootstrap 显著性门控） |
| `DECIDER` | `hybrid` | `rule` \| `llm` \| `hybrid`（默认 hybrid=常态走规则快通路，停滞或爬坡放缓时唤醒 27B，连续未采纳则熔断；无 PyTRiO 端点时自动降级为规则） |
| `PYTRIO_TIMEOUT_S` | `300` | `DECIDER=llm` 时 PyTRiO 请求超时（秒）；27B 冷启动首字节可达 128s~180s+，故默认放大 |
| `SWANLAB` | `1` | 默认开：offline 模式落 `outputs/swanlog/`（挂载卷，`--rm` 后不丢）；置 0 关闭。赛后用 `scripts/swanlab_sync.py` 上传，实验以公开方式创建（C8） |
| `VERIFY` | `1` | 跑完是否做硬约束自检 |

### 启用 27B 慢通路（`DECIDER=hybrid` 推荐，或 `DECIDER=llm`）

上面的一键命令带 `--network none`，LLM 必然连不上而每轮降级。启用慢通路时去掉它，并用 `--env-file` 注入 PyTRiO 凭据（`.env` 已在 `.gitignore`/`.dockerignore` 中，不进仓库也不进镜像）：

```bash
docker run --rm --cpus 4 --memory 8g --env-file .env -e DECIDER=hybrid \
  -v "$PWD/base:/app/base:ro" -v "$PWD/outputs-docker:/app/outputs" zhixun-harness
```

对照实验（6 轮，`scripts/compare_deciders.py`）里 hybrid 只调用 27B 1~3 次，交付值在 B、C 两包为全场最优；纯 `llm` 每轮都调用。

跑完看输出里的 `LLM 慢通路降级 N/M 轮`：N=M 表示 LLM 全程未生效（端点/网络/证书问题）。
可复现说明：请求用 `temperature=0`，但远端大模型不保证逐字一致，LLM 臂的逐轮结果可能有小幅差异；离线规则路径（默认）严格可复现（C11）。

示例：
```bash
# 平台目录名不同
docker run --rm -e BASE_DIR=/data/packages -v /data/packages:/data/packages:ro zhixun-harness

# 只跑 A，5 轮（SwanLab 离线记录默认已开）
docker run --rm -e TASKS=a -e ROUNDS=5 \
  -v "$PWD/base:/app/base:ro" -v "$PWD/outputs-docker:/app/outputs" zhixun-harness
```

缺挂载时不会静默跑空——会打印挂载示例并以退出码 2 退出。

## 4. 宿主机直接运行（不用 Docker）

```bash
python3 -m pip install -r requirements-docker.txt      # 官方路径只需 numpy
python3 scripts/run_official.py --package base/cpu-a-guide-20261110-v1 --rounds 10
python3 scripts/verify_contest_conditions.py           # 硬约束自检
```

自研对照闭环（非评分路径，仅方法探索）需要完整依赖：
```bash
python3 -m pip install -r requirements.txt
python3 scripts/run_loop.py --config configs/a1_gas.yaml
```

## 5. 实测结果

### 容器内（离线、4 核 8G 限制）

| 包 | 轮数 | 官方基线 | 我们最优 | 优于基线 | 闭环 | 耗时 | 峰值内存 |
|---|---|---|---|---|---|---|---|
| A | 3 | 0.7140 | 0.7258 | 是 | 3 | 2.9s | 67MB |
| B | 3 | 681.16 | 666.88 | 是 | 3 | 4.9s | 73MB |
| C | 3 | 52.87 | 52.23 | 是 | 3 | 34.2s | 249MB |

### 宿主机完整 10 轮

| 包 | 官方基线 | 我们最优 | 交付（显著性确认） | 耗时 | 峰值内存 |
|---|---|---|---|---|---|
| A | 0.7140 | 0.7264 | **0.7258**（P=100%） | 6.6s | 59MB |
| B | 681.16 | **666.88** | 666.88（P=100%） | 11.7s | 91MB |
| C | 52.87 | **51.38** | 51.38（P=100%） | 84.8s | 352MB |

### 容器与宿主机逐轮完全一致

同样 3 轮、固定种子 42，A 的逐轮值在容器与宿主机**逐位相同**，可复现（C11）。

## 6. 硬约束自检

```bash
python3 scripts/verify_contest_conditions.py          # 宿主机
# 或容器内自动执行（VERIFY=1 默认开启）
```

自检逐条核对并打印结论：

| 约束 | 实测 |
|---|---|
| C1 无人干预跑完（退出码 0、零降级） | 通过 |
| C2 按时输出（自设预算单包 < 300s，官方未公布时限） | 通过，最慢 85s |
| C3 资源（4 核 8G） | 通过，峰值内存 352MB |
| C4 离线可跑 | 通过（`--network none` 验证） |
| C5 镜像 ≤2GB、不含权重数据 | 通过，402MB |
| C9 每包 ≥3 次完整闭环 | 通过，各 10 条 |
| C10 不低于官方基线 | 通过，三包全优于 |
| C11 一键运行、可复现 | 通过，固定种子两次一致 |

## 7. 产出文件

| 路径 | 内容 |
|---|---|
| `outputs/official_{a,b,c}_rounds.csv` | 逐轮验证指标与上线判定 |
| `outputs/audit_official-{a,b,c}.jsonl` | 审计记录（读历史/假设/实验/决策 + 显著性 p 值与置信区间） |
| `outputs/official/{a,b,c}/round-*/` | 每轮官方 `checkpoint.npz` + `training.json` + `history.jsonl` |
| `outputs/CONCLUSIONS.md` | 全部实验结论 |
| `outputs/swanlog/`（默认 SWANLAB=1） | SwanLab 离线记录，赛后用 `scripts/swanlab_sync.py` 上传 |

**最终交付的 checkpoint** 由日志末尾的"交付 checkpoint"一行给出（显著性确认链优先）。
