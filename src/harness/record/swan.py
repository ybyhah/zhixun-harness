"""SwanLab 封装。

官方口径：SwanLab 的记录对象是**被训练的工业模型**的训练过程，
不是 Harness 自身的调用过程。因此这里只上报工业模型的训练与评估指标。

不可用时自动降级为本地记录，绝不中断闭环（无人干预跑完是排名门槛）。
"""

from __future__ import annotations

from typing import Any


class SwanRecorder:
    def __init__(
        self,
        enabled: bool = False,
        project: str = "zhixun-harness",
        mode: str = "offline",
        config: dict[str, Any] | None = None,
        run_name: str | None = None,
        public: bool = True,
        log_dir: str | None = None,
    ) -> None:
        self.enabled = enabled
        self.mode = mode
        self.public = public          # C8：记录必须公开，上传云端时直接设为公开
        self.degraded = False
        self.degrade_reason = ""
        self._run = None
        self.local_log: list[dict[str, Any]] = []

        if not enabled:
            return
        try:
            import swanlab

            kwargs: dict[str, Any] = dict(
                project=project,
                name=run_name,
                config=config or {},
                mode="offline" if mode == "offline" else "cloud",
                public=public,
            )
            if log_dir:
                kwargs["log_dir"] = str(log_dir)
            self._run = swanlab.init(**kwargs)
        except Exception as exc:  # noqa: BLE001
            self.degraded = True
            self.degrade_reason = f"{type(exc).__name__}: {exc}"
            self._run = None

    @property
    def active(self) -> bool:
        return self._run is not None

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """上报工业模型的训练/评估指标。"""
        payload = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        self.local_log.append({"step": step, **payload})
        if self._run is None:
            return
        try:
            self._run.log(payload, step=step)
        except Exception as exc:  # noqa: BLE001
            self.degraded = True
            self.degrade_reason = f"log 失败 {type(exc).__name__}: {exc}"
            self._run = None

    def finish(self) -> None:
        if self._run is None:
            return
        try:
            self._run.finish()
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._run = None
