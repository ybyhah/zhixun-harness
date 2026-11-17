"""封装一次官方续训：子进程调用 `python -m baselines.train`，解析 training.json。

不改官方逻辑；只拼命令、传路径、读报告。续训约束（新目录、保种子/架构、
B 需 label-provenance）由官方 train.py 自己强制，这里负责正确传参。
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .package import OfficialPackage


@dataclass
class RoundResult:
    stage_id: str
    output_dir: Path
    checkpoint: Path
    report: dict
    ok: bool
    error: str = ""

    @property
    def validation_after(self) -> dict:
        return self.report.get("validation_after", {}) if self.report else {}


def _label_provenance(pkg: OfficialPackage, work: Path) -> Path:
    """B 的标签来源声明：官方 RUL=(N-i) 分钟，确定性计算。"""
    prov = {
        "status": "draft",
        "method": "official RUL = (last_snapshot - current_snapshot) * 1 minute; "
                  "deterministic from full run-to-failure record, no health-stage labels",
    }
    path = work / "label_provenance.json"
    path.write_text(json.dumps(prov, ensure_ascii=False))
    return path


def run_round(
    pkg: OfficialPackage,
    output_dir: Path,
    *,
    resume: Path | None,
    epochs: int,
    stage_id: str,
    learning_rate: float | None = None,
    batch_size: int | None = None,
    timeout_s: float = 600.0,
    cpu_threads: int = 1,
    profile_override: Path | None = None,
    entry_override: Path | None = None,
) -> RoundResult:
    """跑一轮续训。resume=None 时用随包 checkpoint 作 --from-baseline。

    profile_override：自定义 profile JSON（官方 train.py 支持 `--profile *.json`），
    用于加宽实验（只改 hidden，其余与官方 config 一致）。
    entry_override：用指定脚本替代 `-m baselines.train`（改训练损失的运行时注入）。
    """
    output_dir = Path(output_dir).resolve()
    start = (resume or pkg.checkpoint).resolve()
    flag = "--resume" if resume else "--from-baseline"
    profile_arg = str(Path(profile_override).resolve()) if profile_override else pkg.profile
    entry = ([str(Path(entry_override).resolve())] if entry_override
             else ["-m", "baselines.train"])
    cmd = [
        sys.executable, *entry,
        "--profile", profile_arg,
        "--train", str(pkg.train_jsonl),
        "--validation", str(pkg.validation_jsonl),
        flag, str(start),
        "--epochs", str(epochs),
        "--output", str(output_dir),
        "--stage-id", stage_id,
    ]
    if learning_rate is not None:
        cmd += ["--learning-rate", str(learning_rate)]
    if batch_size is not None:
        cmd += ["--batch-size", str(batch_size)]
    work = output_dir.parent
    work.mkdir(parents=True, exist_ok=True)
    if pkg.profile == "B":
        cmd += ["--label-provenance", str(_label_provenance(pkg, work))]

    import os
    env = {**os.environ, "BASELINE_CPU_THREADS": str(cpu_threads)}
    try:
        proc = subprocess.run(
            cmd, cwd=str(pkg.root), capture_output=True, text=True,
            timeout=timeout_s, env=env,
        )
    except subprocess.TimeoutExpired:
        return RoundResult(stage_id, output_dir, output_dir / "checkpoint.npz",
                           {}, ok=False, error=f"训练超时 {timeout_s}s")
    if proc.returncode != 0:
        return RoundResult(stage_id, output_dir, output_dir / "checkpoint.npz",
                           {}, ok=False,
                           error=(proc.stderr.strip().splitlines() or ["非零退出"])[-1])
    report_path = output_dir / "training.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else \
        json.loads(proc.stdout.strip().splitlines()[-1])
    return RoundResult(stage_id, output_dir, output_dir / "checkpoint.npz",
                       report, ok=True)
