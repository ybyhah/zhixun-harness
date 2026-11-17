#!/usr/bin/env python3
"""提交物本地离线自测：describe / train(一轮) / predict，不触真实平台与模型。

- describe：子进程跑 `harness.py describe`，校验协议 JSON 与 tasks。
- train：用 FakeRound 在进程内驱动 contest_submission.train，真实跑官方续训子进程，
  断言本轮有模型调用、产出候选 checkpoint、finish 带非空 model_call_ids。
- predict：子进程跑 `harness.py predict`，用上一步产出的 checkpoint 预测若干样本，校验输出。
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "submission"
SDK = ROOT / "contest-harness-sdk-1.0.2"
HARNESS = SDK / "template" / "harness.py"
PKG_A = ROOT / "base" / "cpu-a-guide-20261110-v1"
sys.path.insert(0, str(SUB))


def _env():
    env = {**os.environ, "CONTEST_SUBMISSION": "contest_submission",
           "PYTHONPATH": os.pathsep.join([str(SUB), str(SDK)]), "BASELINE_CPU_THREADS": "1"}
    return env


def test_describe():
    out = subprocess.run([sys.executable, str(HARNESS), "describe"], cwd=str(SUB),
                         env=_env(), capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    info = json.loads(out.stdout.strip().splitlines()[-1])
    assert info["protocol"] == "contest.harness.workspace.v2", info
    assert set(info["tasks"]) == {"classification", "regression"}, info
    print("[describe] ok:", info["tasks"])


class _Data:
    def __init__(self, pkg):
        self.train_path = str(pkg / "public" / "train.jsonl")
        self.validation_path = str(pkg / "public" / "validation.jsonl")


class _Parent:
    def __init__(self, ckpt):
        self._ckpt = ckpt
        self.artifact_id = "parent-artifact-0"
        self.receipt = {"artifact_id": self.artifact_id}

    def path(self, relative):
        assert relative == "checkpoint.npz", relative
        return Path(self._ckpt)


class FakeRound:
    """进程内替身：只实现 contest_submission.train 用到的接口。"""
    def __init__(self, pkg, work, *, summary=None, history=None, round_index=1):
        self.dataset_key = "A"
        self.task = {"task": "classification", "metric": "macro_f1",
                     "metric_direction": "higher"}
        self.round_index = round_index
        self.max_output_tokens = 512
        self.data = _Data(pkg)
        self.parent = _Parent(pkg / "checkpoint.npz")
        self.call_ids = []
        self.summary = summary or {"macro_f1": 0.80}
        self._history = history or []
        self.submitted = None
        self.finished = None

    def history(self):
        return self._history

    def model(self, messages, **options):
        assert isinstance(messages, list) and messages and "max_tokens" in options
        assert options["response_format"] == {"type": "json_object"}
        assert options["max_tokens"] <= 256
        self.call_ids.append("call-local-1")

        class R:
            call_id = "call-local-1"
            text = '{"epochs": 2, "learning_rate": 0.003, "hypothesis": "小步续训看是否稳升"}'
        return R()

    def submit_candidate(self, directory, **kw):
        assert (Path(directory) / "checkpoint.npz").is_file(), "候选缺少 checkpoint"
        assert kw["checkpoint_paths"] == ["checkpoint.npz"] and kw["predict_argv"][-1] == "predict"
        self.submitted = {"artifact_id": "artifact-local-1", **kw}
        return {"artifact_id": "artifact-local-1", "summary": {"metrics": self.summary}}

    def finish(self, **kw):
        assert kw["model_call_ids"], "finish 的 model_call_ids 不能为空"
        self.finished = kw


def _run_train(decide_hook, *, summary=None, history=None, round_index=1):
    """跑一轮 train，用 decide_hook 替换门控，返回 (rnd, cand_ckpt, report)。"""
    import contest_submission
    tmp = tempfile.mkdtemp(prefix="sdk-local-")
    os.environ["CONTEST_WORK"] = tmp
    rnd = FakeRound(PKG_A, Path(tmp), summary=summary, history=history,
                    round_index=round_index)
    orig = contest_submission._decide_promote
    contest_submission._decide_promote = decide_hook
    try:
        contest_submission.train(rnd)
    finally:
        contest_submission._decide_promote = orig
    assert rnd.call_ids == ["call-local-1"], "本轮必须有一次模型调用"
    assert rnd.submitted and rnd.finished, "必须 submit + finish"
    ckpt = Path(tmp) / "cand" / "checkpoint.npz"
    assert ckpt.is_file(), "续训未产出 checkpoint"
    report = json.loads((Path(tmp) / "cand" / "training.json").read_text())
    return rnd, str(ckpt), report


def test_train_promote():
    rnd, ckpt, _ = _run_train(lambda *a, **k: (True, {"p_better": 1.0}))
    assert rnd.finished["selected_artifact_id"] == "artifact-local-1", "晋升应选候选"
    print("[train/promote] ok: 选中候选，model_call_ids=", rnd.finished["model_call_ids"])
    return ckpt


def test_train_keep_parent():
    history = [{"id": "round-1", "round_index": 1,
                "selected_artifact_id": "parent-artifact-0",
                "summary": {"metrics": {"macro_f1": 0.80}},
                "configuration": {"epochs": 5, "learning_rate": 0.001}}]
    rnd, _, _ = _run_train(
        lambda *a, **k: (True, {"p_better": 1.0}),
        summary={"macro_f1": 0.79}, history=history, round_index=2)
    assert rnd.finished["selected_artifact_id"] == "parent-artifact-0", "不显著应保留父模型"
    assert "保留父模型" in rnd.finished["reason"]
    print("[train/keep] ok: 保留父模型，reason=", rnd.finished["reason"][:40], "…")


def test_train_fail_keep_parent():
    """续训子进程失败 → 不提交候选、优雅保留父模型、本轮正常 finish。"""
    import contest_submission
    tmp = tempfile.mkdtemp(prefix="sdk-fail-")
    (Path(tmp) / "cand").mkdir()          # 预占输出目录 → 官方 train 拒绝写已存在目录 → 子进程失败
    os.environ["CONTEST_WORK"] = tmp
    rnd = FakeRound(PKG_A, Path(tmp))
    contest_submission.train(rnd)
    assert rnd.call_ids == ["call-local-1"], "失败路径也必须已调用模型"
    assert rnd.submitted is None, "训练失败不应提交候选"
    assert rnd.finished and rnd.finished["selected_artifact_id"] == "parent-artifact-0"
    assert "train_failed" in rnd.finished["configuration"]
    print("[train/fail] ok: 续训失败→保留父模型，model_call_ids=", rnd.finished["model_call_ids"])


def test_real_decision():
    """在真实父/候选检查点上跑真实门控，确认不抛错、返回 (bool, dict)。"""
    import contest_submission
    rnd, cand_ckpt, report = _run_train(lambda *a, **k: (True, {}))
    promote, info = contest_submission._decide_promote(
        rnd, "classification", rnd.parent.path("checkpoint.npz"), Path(cand_ckpt), report)
    assert isinstance(promote, bool) and isinstance(info, dict), (promote, info)
    print("[decision/real] ok: promote=", promote, "info=", info)


def test_decision_unit():
    sys.path.insert(0, str(SUB))
    import numpy as np
    from decision import should_promote
    import importlib
    importlib.import_module("decision")
    y = np.array([0, 1, 2, 0, 1, 2])
    correct = np.eye(3)[y]
    wrong = np.eye(3)[(y + 1) % 3]
    p_ok, info = should_promote(wrong, correct, y, "classification", bootstrap=200)
    assert p_ok is True, (p_ok, info)
    p_same, _ = should_promote(correct, correct, y, "classification", bootstrap=200)
    assert p_same is False, p_same
    print("[decision/unit] ok: 全对>全错 显著；相同不晋升")


def test_predict(ckpt):
    samples = []
    for line in Path(PKG_A, "public", "validation.jsonl").read_text().splitlines()[:3]:
        row = json.loads(line)
        samples.append(json.dumps({"task": "classification", "inputs": row["inputs"]}))
    env = {**_env(), "CONTEST_ARTIFACT_PATH": ckpt,
           "CONTEST_DATASET_KEY": "A", "CONTEST_TASK": "classification"}
    out = subprocess.run([sys.executable, str(HARNESS), "predict"], cwd=str(SUB), env=env,
                         input="\n".join(samples) + "\n", capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    rows = [json.loads(l) for l in out.stdout.splitlines() if l.strip()]
    assert len(rows) == len(samples), (len(rows), out.stderr)
    for r in rows:
        outs = r["outputs"]
        assert isinstance(outs, list) and sum(o["score"] == max(x["score"] for x in outs) for o in outs) == 1
    print("[predict] ok:", len(rows), "行，示例 top=",
          max(rows[0]["outputs"], key=lambda o: o["score"])["label"])


if __name__ == "__main__":
    test_describe()
    test_decision_unit()
    ckpt = test_train_promote()
    test_train_keep_parent()
    test_train_fail_keep_parent()
    test_real_decision()
    test_predict(ckpt)
    print("\n全部本地自测通过。")


