#!/usr/bin/env bash
# 一键运行：驱动官方 base/ 包续训，跑完后自动做赛题硬约束自检。
#
# 用法（宿主机）：
#   docker run --rm --network none \
#     -v "$PWD/base:/app/base:ro" -v "$PWD/outputs-docker:/app/outputs" \
#     zhixun-harness
#
# 环境变量：
#   BASE_DIR  官方包目录（默认 /app/base）
#   TASKS     要跑的任务，逗号分隔，如 a,c（默认 a,b,c）
#   ROUNDS    每包续训轮数（默认 10）
#   GATE      plain | significance（默认 significance）
#   DECIDER   rule | llm | hybrid（默认 hybrid；无 PyTRiO 端点时自动降级为规则，等价 rule）
#   SWANLAB   0 | 1（默认 1；offline 模式落 outputs/swanlog/，赛后 swanlab_sync.py 上传并公开）
#   VERIFY    0 | 1（默认 1；跑完做硬约束自检）
# 额外命令行参数会原样透传给 scripts/run_official.py。

set -uo pipefail

BASE_DIR="${BASE_DIR:-/app/base}"
TASKS="${TASKS:-a,b,c}"
ROUNDS="${ROUNDS:-10}"
GATE="${GATE:-significance}"
DECIDER="${DECIDER:-hybrid}"
SWANLAB="${SWANLAB:-1}"
VERIFY="${VERIFY:-1}"
OUT_DIR="/app/outputs"

if [ ! -d "$BASE_DIR" ] || [ -z "$(ls -d "$BASE_DIR"/cpu-* 2>/dev/null)" ]; then
  cat >&2 <<EOF
错误：未找到官方包目录，期望在 $BASE_DIR 下有 cpu-a-* / cpu-b-* / cpu-c-* 。

镜像刻意不打包 base/（含官方工业模型权重与公开数据），需运行时挂载：

  docker run --rm --network none \\
    -v "\$PWD/base:/app/base:ro" \\
    -v "\$PWD/outputs-docker:/app/outputs" \\
    zhixun-harness

若平台的固定目录名不同，用 BASE_DIR 覆盖：
  docker run --rm -e BASE_DIR=/data/packages -v /data/packages:/data/packages:ro zhixun-harness
EOF
  exit 2
fi

mkdir -p "$OUT_DIR" 2>/dev/null || true
if ! touch "$OUT_DIR/.writable" 2>/dev/null; then
  echo "提示：$OUT_DIR 不可写，结果将留在容器内。建议挂载：-v \"\$PWD/outputs-docker:/app/outputs\"" >&2
else
  rm -f "$OUT_DIR/.writable"
fi

SWAN_FLAG=()
[ "$SWANLAB" = "1" ] && SWAN_FLAG=(--swanlab --swanlab-mode offline)

echo "=================================================================="
echo "智训引擎 Harness 一键运行"
echo "  官方包目录 : $BASE_DIR"
echo "  任务       : $TASKS   轮数: $ROUNDS"
echo "  门控       : $GATE    决策器: $DECIDER   SwanLab: $SWANLAB"
echo "  产出目录   : $OUT_DIR"
echo "=================================================================="

status=0
IFS=',' read -ra WANTED <<< "$TASKS"
for t in "${WANTED[@]}"; do
  t="$(echo "$t" | tr -d '[:space:]' | tr 'A-Z' 'a-z')"
  pkg="$(ls -d "$BASE_DIR"/cpu-"$t"-* 2>/dev/null | head -1)"
  if [ -z "$pkg" ]; then
    echo "跳过任务 $t：$BASE_DIR 下没有对应的 cpu-$t-* 包" >&2
    continue
  fi
  echo ""
  echo "---------- 任务 ${t^^}：$(basename "$pkg") ----------"
  python3 /app/scripts/run_official.py \
    --package "$pkg" --rounds "$ROUNDS" \
    --gate "$GATE" --decider "$DECIDER" "${SWAN_FLAG[@]}" "$@"
  rc=$?
  [ $rc -ne 0 ] && status=$rc && echo "任务 ${t^^} 以退出码 $rc 结束" >&2
done

if [ "$VERIFY" = "1" ]; then
  echo ""
  echo "=================================================================="
  echo "赛题硬约束自检"
  echo "=================================================================="
  python3 /app/scripts/verify_contest_conditions.py --rounds "$ROUNDS" || status=$?
fi

echo ""
echo "完成（退出码 $status）。产出在 $OUT_DIR"
exit $status
