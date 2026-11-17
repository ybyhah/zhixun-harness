import json

import numpy as np

from harness.record.audit import AuditLog
from harness.record.schema import DecisionRecord
from harness.record.swan import SwanRecorder


def rec(i=0, complete=True):
    return DecisionRecord(
        batch_index=i,
        trigger="性能下降",
        drift_type="CONCEPT",
        evidence={"psi_mean": np.float64(0.3), "perf": np.float32(0.7)},
        history_read=[0, 1] if complete else [],
        hypothesis="窗口过短导致信息丢失" if complete else None,
        experiments=[{"name": "window=3", "score": 0.8}] if complete else [],
        decision={"action": "WINDOW_RETRAIN"} if complete else {},
        gate_result={"accepted": True},
        cost={"trainings": 1},
    )


# ---------- schema ----------

def test_complete_loop_requires_all_four_fields():
    assert rec(complete=True).is_complete_loop()
    assert not rec(complete=False).is_complete_loop()


def test_partial_loop_is_incomplete():
    r = rec()
    r.hypothesis = None
    assert not r.is_complete_loop()


def test_roundtrip_through_dict():
    r = rec(3)
    back = DecisionRecord.from_dict(r.to_dict())
    assert back.batch_index == 3 and back.drift_type == "CONCEPT"


def test_from_dict_ignores_unknown_keys():
    d = rec().to_dict()
    d["未来字段"] = 1
    assert DecisionRecord.from_dict(d).batch_index == 0


# ---------- audit ----------

def test_append_and_read_back(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl")
    log.append(rec(0))
    log.append(rec(1))
    assert len(log) == 2
    assert [r.batch_index for r in AuditLog.read(tmp_path / "a.jsonl")] == [0, 1]


def test_numpy_values_are_serializable(tmp_path):
    path = tmp_path / "a.jsonl"
    AuditLog(path).append(rec())
    payload = json.loads(path.read_text().splitlines()[0])
    assert isinstance(payload["evidence"]["psi_mean"], float)


def test_nan_becomes_null(tmp_path):
    path = tmp_path / "a.jsonl"
    log = AuditLog(path)
    r = rec()
    r.evidence["score"] = float("nan")
    log.append(r)
    payload = json.loads(path.read_text().splitlines()[0])
    assert payload["evidence"]["score"] is None


def test_backfill_outcome(tmp_path):
    path = tmp_path / "a.jsonl"
    log = AuditLog(path)
    log.append(rec(5))
    assert log.backfill_outcome(5, {"next_batch_score": 0.82})
    assert not log.backfill_outcome(99, {"x": 1})
    assert AuditLog.read(path)[0].outcome == {"next_batch_score": 0.82}


def test_complete_loops_filter(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl")
    log.append(rec(0, complete=True))
    log.append(rec(1, complete=False))
    log.append(rec(2, complete=True))
    assert [r.batch_index for r in log.complete_loops()] == [0, 2]


def test_reset_clears_previous_file(tmp_path):
    path = tmp_path / "a.jsonl"
    AuditLog(path).append(rec(0))
    fresh = AuditLog(path, reset=True)
    assert len(fresh) == 0 and AuditLog.read(path) == []


def test_read_missing_file_returns_empty(tmp_path):
    assert AuditLog.read(tmp_path / "nope.jsonl") == []


# ---------- swanlab ----------

def test_disabled_recorder_still_logs_locally():
    r = SwanRecorder(enabled=False)
    r.log({"macro_f1": 0.9, "ignored": "text"}, step=1)
    assert not r.active and not r.degraded
    assert r.local_log == [{"step": 1, "macro_f1": 0.9}]
    r.finish()


def test_recorder_degrades_when_init_fails(monkeypatch):
    """SwanLab 不可用时必须降级而不是抛异常，否则闭环会中断。"""
    import builtins

    real_import = builtins.__import__

    def boom(name, *args, **kwargs):
        if name == "swanlab":
            raise RuntimeError("无网络")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", boom)
    r = SwanRecorder(enabled=True)
    assert r.degraded and not r.active
    r.log({"macro_f1": 0.5})
    assert r.local_log[-1]["macro_f1"] == 0.5


def test_recorder_offline_mode_creates_run(tmp_path, monkeypatch):
    """offline 模式应真正建 run、active 为真、log 不崩、finish 正常。"""
    monkeypatch.chdir(tmp_path)
    r = SwanRecorder(enabled=True, mode="offline", project="test-harness",
                     run_name="a1-full", config={"task": "a1", "seed": 42})
    assert r.active and not r.degraded
    assert r.public is True
    r.log({"macro_f1": 0.8, "score": 0.8, "note": "skip-nonnumeric"}, step=0)
    r.log({"macro_f1": 0.6}, step=1)
    assert r.local_log[0] == {"step": 0, "macro_f1": 0.8, "score": 0.8}
    r.finish()
    assert not r.active


def test_recorder_public_flag_default():
    assert SwanRecorder(enabled=False).public is True
    assert SwanRecorder(enabled=False, public=False).public is False
