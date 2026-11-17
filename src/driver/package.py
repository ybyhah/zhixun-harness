"""官方 base/ 包的定位与元信息。

每个包形如 base/cpu-{a,b,c}-...-vN，内含：
  baseline-info.json  profile/task/parameter_count/checkpoint 校验和
  baselines/          官方 train.py/model.py/data.py/predict.py/configs
  checkpoint.npz      初始权重（续训起点）
  public/train.jsonl, public/validation.jsonl
本模块只读取元信息并给出路径，不触碰官方训练逻辑。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class OfficialPackage:
    root: Path
    profile: str          # A | B | C
    task: str             # classification | regression
    parameter_count: int
    checkpoint: Path
    train_jsonl: Path
    validation_jsonl: Path

    @property
    def metric_name(self) -> str:
        return "macro_f1" if self.task == "classification" else "rmse"

    @property
    def higher_better(self) -> bool:
        return self.task == "classification"

    def config_path(self) -> Path:
        return self.root / "baselines" / "configs" / f"{self.profile}.json"


def load_package(root: str | Path, verify_checkpoint: bool = True) -> OfficialPackage:
    root = Path(root).resolve()
    info = json.loads((root / "baseline-info.json").read_text())
    profile = info.get("dataset_key")
    if profile not in ("A", "B", "C"):
        raise ValueError(f"未知 dataset_key: {profile!r}")
    task = info["task"]
    checkpoint = root / "checkpoint.npz"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"缺少 checkpoint.npz: {checkpoint}")
    declared = info.get("checkpoint_sha256")
    if verify_checkpoint and declared and _sha256(checkpoint) != declared:
        raise ValueError(
            f"checkpoint.npz 校验和与 baseline-info.json 不符：{checkpoint}"
        )
    train = root / "public" / "train.jsonl"
    val = root / "public" / "validation.jsonl"
    for p in (train, val):
        if not p.is_file():
            raise FileNotFoundError(f"缺少数据文件: {p}")
    return OfficialPackage(
        root=root, profile=profile, task=task,
        parameter_count=int(info.get("parameter_count", 0)),
        checkpoint=checkpoint, train_jsonl=train, validation_jsonl=val,
    )
