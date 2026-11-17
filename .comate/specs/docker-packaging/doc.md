# Docker 一键运行打包

> 补齐 C5（镜像 ≤2GB、不打包大模型权重）与 C11（一键运行、可复现），并保持 C4（离线）、C3（4 核 8G）。
> 不改任何算法逻辑，只做打包与入口。

## 0. 关键取舍（先想清楚）

### 0.1 镜像只装运行官方评分路径所必需的依赖

实测：
- 官方包 `baselines/requirements.txt` 只要 **numpy==1.26.4**
- 我们的官方驱动（`src/driver/` + `scripts/run_official.py`）第三方依赖只有 **numpy**（其余全是标准库）
- `pandas/scikit-learn/scipy/matplotlib/frouros` 只被**自研对照闭环**（`src/harness/`）使用，不在评分路径上

因此镜像分两层依赖：
- **必装**：`numpy`（官方路径）、`swanlab`（实验记录，离线模式也要它）
- **不装**：pandas / sklearn / scipy / matplotlib / frouros / pytest / river / nannyml

这样镜像能压到很小，远低于 2GB，也减少离线环境的装包风险。自研对照不进镜像——它本来就只是本地方法探索。

### 0.2 `base/` 不打进镜像，运行时挂载

- `base/` 共 58MB，含官方 `checkpoint.npz`（工业模型权重）与 `public/*.jsonl`。
- 赛题 C5 的"严禁打包模型权重"主要约束**训练师模型（大模型）权重**，工业模型权重按 `[逐字稿]` 是"不推荐但未禁止"。
- 更重要的是官方说法：数据包"上传至指定平台固定目录下"——**平台会自己提供**。
- 所以镜像**不含** `base/`，运行时用 `-v` 挂载到 `/app/base`。入口脚本检测不到就明确报错并提示挂载方式，而不是静默跑空。

### 0.3 不打包任何数据与第三方仓库

`.dockerignore` 排除 `base/`、`data/`、`outputs/`、`backup/`、`third_party/`、`swanlog/`、`__pycache__`、`.pytest_cache`、`*.rar`、`*.npz`。

## 1. 文件设计

### `Dockerfile`
```
FROM python:3.11-slim
WORKDIR /app
# 只装官方路径必需的依赖（见 0.1）
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY run.sh .
ENV PYTHONHASHSEED=0 BASELINE_CPU_THREADS=1 PYTHONPATH=/app/src
ENTRYPOINT ["/app/run.sh"]
```
新增 `requirements-docker.txt`：只含 `numpy==1.26.4`（与官方包锁定一致）、`swanlab>=0.10`。

### `run.sh`（一键入口）
- 先校验 `/app/base` 存在且含至少一个 `cpu-*` 包；缺失则打印挂载示例并以非零退出。
- 默认依次跑 A、B、C 全部 10 轮；可用环境变量覆盖：
  `TASKS`（如 `a,c`）、`ROUNDS`、`GATE`（plain/significance）、`DECIDER`（rule/llm）、`SWANLAB`（0/1）。
- 结果写 `/app/outputs`（建议挂载出来）。
- 末尾自动跑 `scripts/verify_contest_conditions.py` 做硬约束自检，把结论打在日志最后。
- 透传额外参数：`docker run <img> --rounds 3` 之类直接转给 `run_official.py`。

### `.dockerignore`
如 0.3 所列。

## 2. 影响文件

| 文件 | 改动 |
|---|---|
| `Dockerfile` | 新增 |
| `run.sh` | 新增，可执行 |
| `.dockerignore` | 新增 |
| `requirements-docker.txt` | 新增（镜像内最小依赖） |
| `requirements.txt` | 加注释说明"镜像只用 requirements-docker.txt" |
| `scripts/verify_contest_conditions.py` | 无需改（已支持 `--quick`，run.sh 调用它） |

不改 `src/`、`base/`、算法与配置。

## 3. 验证

1. `docker build -t zhixun-harness .` 成功（需联网拉基础镜像与装包，申请一次升级权限）。
2. `docker images` 确认镜像 **≤2GB**（预期 ~250–400MB）。
3. 冒烟：
   ```
   docker run --rm -v $PWD/base:/app/base:ro -v $PWD/outputs-docker:/app/outputs \
     zhixun-harness
   ```
   确认三包跑通、结果与宿主机一致（固定种子）、自检全过。
4. 缺挂载时的行为：`docker run --rm zhixun-harness` 应给出清晰报错与挂载示例，非零退出。
5. 离线验证：`docker run --network none ...` 必须照样跑完（C4）。
6. 资源：`--cpus 4 --memory 8g` 下跑通（C3）。
7. 宿主机 `python3 -m pytest` 仍全绿。

## 4. 边界与风险

| 情况 | 处理 |
|---|---|
| 基础镜像拉取需联网 | build 时申请一次升级权限；失败则给出手动命令 |
| `base/` 未挂载 | 明确报错 + 挂载示例，非零退出（不静默跑空） |
| `outputs` 未挂载 | 容器内仍能跑，结果留在容器内；日志提示建议挂载 |
| 平台固定目录名不同 | `BASE_DIR` 环境变量可覆盖 `/app/base` |
| swanlab 在离线环境初始化失败 | `SwanRecorder` 已有降级逻辑，且默认 `SWANLAB=0` 不启用 |
| 镜像意外超 2GB | 逐项裁依赖；必要时改 `python:3.11-alpine`（但 numpy 编译成本高，优先 slim） |

## 5. 预期产出

1. 可构建的镜像，≤2GB，不含 `base/`、数据、第三方仓库。
2. `run.sh` 一条命令跑完三包并附硬约束自检结论。
3. 复现文档片段（build / run / 挂载 / 离线 / 资源限制的完整命令）。
