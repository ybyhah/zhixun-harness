# Harness 镜像：装 SDK + 官方 baselines + 选手提交模块。不含密钥、不含 base/ 权重。
FROM python:3.12-slim-bookworm

RUN pip install --no-cache-dir numpy==1.26.4

# 安装 SDK（纯 Python、无外部依赖）
COPY contest-harness-sdk-1.0.2 /opt/contest-harness-sdk
RUN pip install --no-cache-dir /opt/contest-harness-sdk

# 选手代码：官方 baselines（训练/预测，含 A/B/C 配置）+ 提交模块 + SDK 入口模板
COPY submission/baselines /app/baselines
COPY submission/contest_submission.py /app/contest_submission.py
COPY submission/decision.py /app/decision.py
COPY contest-harness-sdk-1.0.2/template/harness.py /app/harness.py

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 BASELINE_CPU_THREADS=1
WORKDIR /app
# 非 root 运行；/app 只读，工作目录由平台挂载为 /workspace
USER 65534:65534
ENTRYPOINT ["python3", "/app/harness.py"]
