# Docker 一键运行打包任务计划

> 对应 `.comate/specs/docker-packaging/doc.md`。
> 红线：镜像 ≤2GB；不打包 `base/`、数据、第三方仓库；离线可跑（C4）；缺挂载要明确报错而非静默跑空；
> 不改 `src/`、`base/`、算法与配置。

- [x] Task 1: 镜像内最小依赖
    - 1.1: 新增 `requirements-docker.txt`：`numpy==1.26.4`（与官方包锁定一致）+ `swanlab>=0.10`
    - 1.2: `requirements.txt` 顶部加注释说明"镜像只用 requirements-docker.txt；pandas/sklearn/scipy/matplotlib/frouros 仅自研对照需要"
    - 1.3: 本机验证：仅用这两个依赖能跑通 `scripts/run_official.py`（临时隔离验证，不污染现有环境）

- [x] Task 2: .dockerignore
    - 2.1: 新增 `.dockerignore`，排除 `base/`、`data/`、`outputs/`、`backup/`、`third_party/`、`swanlog/`、`__pycache__/`、`.pytest_cache/`、`*.rar`、`*.npz`、`.git/`
    - 2.2: 用 `docker build --no-cache` 的构建上下文大小确认排除生效（上下文应只有几百 KB）

- [x] Task 3: Dockerfile
    - 3.1: `python:3.11-slim` 基础镜像；`WORKDIR /app`
    - 3.2: 先 COPY 依赖文件再装包（利用层缓存），`pip install --no-cache-dir`
    - 3.3: COPY `src/ scripts/ configs/ run.sh`
    - 3.4: `ENV PYTHONHASHSEED=0 BASELINE_CPU_THREADS=1 PYTHONPATH=/app/src`
    - 3.5: `ENTRYPOINT ["/app/run.sh"]`

- [x] Task 4: run.sh 一键入口
    - 4.1: 校验 `${BASE_DIR:-/app/base}` 存在且含至少一个 `cpu-*` 包；缺失则打印挂载示例并非零退出
    - 4.2: 默认依次跑 A/B/C 各 10 轮；环境变量可覆盖 `TASKS`、`ROUNDS`、`GATE`、`DECIDER`、`SWANLAB`
    - 4.3: 结果写 `/app/outputs`；未挂载时提示建议挂载
    - 4.4: 末尾自动跑 `scripts/verify_contest_conditions.py` 并把自检结论打在日志最后
    - 4.5: 透传额外命令行参数给 `run_official.py`
    - 4.6: `chmod +x run.sh`

- [x] Task 5: 构建与体积核验
    - 5.1: `docker build -t zhixun-harness .`（需联网，申请一次升级权限）
    - 5.2: `docker images` 确认 ≤2GB 并记录实际大小
    - 5.3: 确认镜像内不存在 `base/`、`data/`、`third_party/`（`docker run --entrypoint ls` 核查）

- [x] Task 6: 运行验证
    - 6.1: 挂载 `base/` 与 `outputs/` 跑通三包，结果与宿主机逐轮一致（固定种子）
    - 6.2: 不挂载 `base/` 时给出清晰报错与挂载示例、非零退出
    - 6.3: `--network none` 离线跑通（C4）
    - 6.4: `--cpus 4 --memory 8g` 跑通并记录耗时与峰值内存（C3）
    - 6.5: 容器内自检脚本全项通过

- [x] Task 7: 文档与收尾
    - 7.1: 在 `outputs/` 或 `backup/README.md` 之外新增一份运行说明（build / run / 挂载 / 离线 / 资源限制完整命令）
    - 7.2: 宿主机 `python3 -m pytest` 仍全绿；官方驱动产出未被破坏
    - 7.3: 更新 `CONCLUSIONS.md` 或提交物清单，标注 C5/C11 已满足
