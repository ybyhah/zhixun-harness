"""按配置构造任务适配器。新增任务在此登记即可。"""

from __future__ import annotations

from pathlib import Path


def build_adapter(cfg, root: Path):
    """根据 cfg.task.adapter 返回对应的 TaskAdapter。"""
    name = cfg.task.get("adapter", "gas_sensor")

    if name == "xjtu":
        # 读特征缓存而非原始目录；官方若提供预提取特征，在此改读官方文件
        from .xjtu import XJTUAdapter

        return XJTUAdapter(
            root / cfg.data["cache"],
            stream_mode=cfg.data.get("stream_mode", "condition"),
            single_condition=int(cfg.data.get("single_condition", 0)),
            severe_frac=float(cfg.data.get("severe_frac", 0.9)),
            fallback_fpt=float(cfg.data.get("fallback_fpt", 0.6)),
            severe_after_fpt=bool(cfg.data.get("severe_after_fpt", True)),
            fpt_kwargs=dict(cfg.data.get("fpt", {}) or {}),
        )

    raw_dir = root / cfg.data["raw_dir"]

    if name == "gas_sensor":
        from .gas_sensor import GasSensorAdapter

        return GasSensorAdapter(raw_dir, cfg.data.get("n_batches", 10))

    if name == "cmapss":
        from .cmapss import CMAPSSAdapter

        return CMAPSSAdapter(
            raw_dir,
            mode=cfg.data.get("stream_mode", "progressive"),
            rul_cap=float(cfg.data.get("rul_cap", 125)),
            batches_per_subset=int(cfg.data.get("batches_per_subset", 3)),
            single_subset=cfg.data.get("single_subset", "FD001"),
            n_batches=int(cfg.data.get("n_batches", 10)),
            drop_constant_sensors=bool(cfg.data.get("drop_constant_sensors", True)),
            seed=cfg.seed,
        )

    raise ValueError(f"未知的 adapter: {name}")
