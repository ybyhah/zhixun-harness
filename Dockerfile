# 智训引擎 Harness —— 驱动官方 base/ 包续训的一键运行镜像
#
# 设计要点：
#   - 只装官方评分路径必需的依赖（numpy + 可选 swanlab），镜像远小于 2GB（C5）
#   - 不打包 base/（官方包含工业模型权重与数据），运行时挂载（见 run.sh 提示）
#   - 离线可跑（C4）：不在运行期联网；SwanLab 默认关闭，开启时也可用 offline 模式
#   - 固定 PYTHONHASHSEED 与 CPU 线程数，保证可复现（C11）

FROM python:3.11-slim

WORKDIR /app

# 先装依赖，利用层缓存
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# 再拷代码（base/、data/、outputs/ 已在 .dockerignore 排除）
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY run.sh .
RUN chmod +x /app/run.sh

ENV PYTHONHASHSEED=0 \
    BASELINE_CPU_THREADS=1 \
    PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["/app/run.sh"]
