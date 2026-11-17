"""审计日志：JSONL 追加写。

每行一条 DecisionRecord。选 JSONL 是因为追加便宜、容易按行回放，
并且进程被中断也不会损坏已写入的部分。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import DecisionRecord


def _json_safe(obj: Any) -> Any:
    """numpy 标量与数组不能直接序列化，这里统一转原生类型。"""
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if value != value else value
    if isinstance(obj, float) and obj != obj:   # NaN 转 None，保证合法 JSON
        return None
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


class AuditLog:
    def __init__(self, path: str | Path, reset: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if reset and self.path.exists():
            self.path.unlink()
        self._records: list[DecisionRecord] = []

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> list[DecisionRecord]:
        return list(self._records)

    def append(self, record: DecisionRecord) -> None:
        self._records.append(record)
        self._flush()

    def backfill_outcome(self, batch_index: int, outcome: dict) -> bool:
        """给指定批次的记录回填实际结果，用于复盘。"""
        for rec in self._records:
            if rec.batch_index == batch_index:
                rec.outcome.update(outcome)
                self._flush()
                return True
        return False

    def complete_loops(self) -> list[DecisionRecord]:
        return [r for r in self._records if r.is_complete_loop()]

    def _flush(self) -> None:
        """全量重写。记录条数等于批次数，量很小，换取 outcome 可回填。"""
        with self.path.open("w", encoding="utf-8") as fh:
            for rec in self._records:
                fh.write(json.dumps(_json_safe(rec.to_dict()), ensure_ascii=False) + "\n")

    @staticmethod
    def read(path: str | Path) -> list[DecisionRecord]:
        p = Path(path)
        if not p.exists():
            return []
        out: list[DecisionRecord] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(DecisionRecord.from_dict(json.loads(line)))
        return out
