#!/usr/bin/env python3
"""本地代理测量：提交物单轮（公开验证投影）相对官方基线的交付增益 + 门控决策。

警告：这是「批次1→批次2 单步 + 桩掉的模型」代理，不是平台隐藏漂移流的真实分数。
仅用于确认提交物在本地按合约产出、门控是否只采纳显著增益。
"""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "submission"
sys.path.insert(0, str(SUB))
import contest_submission as cs  # noqa: E402

PKGS = [("A", "classification", "macro_f1", True, "base/cpu-a-guide-20261110-v1"),
        ("B", "regression", "rmse", False, "base/cpu-b-rul-20261110-v1"),
        ("C", "regression", "rmse", False, "base/cpu-c-guide-20261110-v1")]


class _Data:
    def __init__(self, pkg):
        self.train_path = str(pkg / "public" / "train.jsonl")
        self.validation_path = str(pkg / "public" / "validation.jsonl")


class _Parent:
    def __init__(self, ckpt):
        self._ckpt, self.artifact_id = ckpt, "parent-0"

    def path(self, rel):
        return Path(self._ckpt)


class FR:
    def __init__(self, key, task, higher, pkg):
        self.dataset_key, self.round_index, self.max_output_tokens = key, 1, 512
        self.task = {"task": task, "metric": "macro_f1" if task == "classification" else "rmse",
                     "higher_is_better": higher}
        self.data, self.parent = _Data(pkg), _Parent(pkg / "checkpoint.npz")
        self.call_ids, self.finished = [], None

    def history(self):
        return []

    def model(self, messages, **o):
        self.call_ids.append("c1")
        return type("R", (), {"call_id": "c1",
                              "text": '{"epochs":5,"learning_rate":0.003,"hypothesis":"官方默认起点续训一轮"}'})()

    def submit_candidate(self, d, **kw):
        return {"artifact_id": "cand-1", "summary": {}}

    def finish(self, **kw):
        self.finished = kw


def main():
    print(f"{'包':<3}{'指标':<9}{'基线(不训)':>12}{'交付':>12}{'增益':>10}  决策")
    for key, task, metric, higher, rel in PKGS:
        pkg = ROOT / rel
        tmp = tempfile.mkdtemp(prefix=f"measure-{key}-")
        os.environ["CONTEST_WORK"] = tmp
        rnd = FR(key, task, higher, pkg)
        cs.train(rnd)
        report = json.loads((Path(tmp) / "cand" / "training.json").read_text())
        before = report["validation_before"][metric]
        after = report["validation_after"][metric]
        promoted = rnd.finished["selected_artifact_id"] == "cand-1"
        delivered = after if promoted else before
        gain = (delivered - before) if higher else (before - delivered)
        dec = rnd.finished["configuration"]["decision"]
        p = dec.get("p_better", dec.get("fallback", "-"))
        print(f"{key:<3}{metric:<9}{before:>12.4f}{delivered:>12.4f}{gain:>+10.4f}  "
              f"{'采纳候选' if promoted else '保留基线'} p_better={p}")


if __name__ == "__main__":
    main()
