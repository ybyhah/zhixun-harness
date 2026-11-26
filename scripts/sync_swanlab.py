"""把三次正式评测记录同步到 SwanLab（需先在终端运行 `swanlab login`）。
用法：python scripts/sync_swanlab.py [--workspace Annie] [--project zhixun-harness-official]

- 11-25（54.84 分）：从评测文档的 A/B/C 逐轮表格解析平台指标，逐轮 log。
- 11-22（49 分）、11-24（42 分）：文档只有分项得分，没有逐轮指标，只记录分项得分。
数字全部来自 reports/formal_evaluation_results/ 下的文档，脚本不另算分数。
"""
import argparse
import re
from pathlib import Path

import swanlab

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "reports" / "formal_evaluation_results"

RUNS = [
    {"name": "formal-20261122-49", "doc": "镜像_2026-11-22_正式评测结果_49分基准交接文档.md",
     "image": "zhixun-submit:linux-amd64-20261122", "total": 49,
     "scores": {"objective": 11.608652, "avg_improvement": 2.640398, "worst_stage": 0,
                "recovery": 3.968254, "stability": 5, "resource_efficiency": 0}},
    {"name": "formal-20261124-42-teammate", "doc": "formal_evaluation_42_2026-11-26.md",
     "image": "zhixun-submit:linux-amd64-20261124", "total": 42,
     "scores": {"objective": 11.709675, "avg_improvement": 2.741421, "worst_stage": 0,
                "recovery": 3.968254, "stability": 5, "resource_efficiency": 0}},
    {"name": "formal-20261125-54.84-best", "doc": "镜像_2026-11-25_正式评测结果_当前最优54.84分.md",
     "image": "zhixun-submit:harness-20261125",
     "tar_sha256": "544055e9fc065adc520cd18be96971e69811ab1cc6cba7ec6c5e75e7b510c892", "total": 54.84,
     "scores": {"objective": 11.677111, "avg_improvement": 2.975956, "worst_stage": 0.090044,
                "recovery": 3.611111, "stability": 5, "resource_efficiency": 0}},
]

# 11-25 文档各包表头（列顺序）→ 指标名
SECTIONS = {"A": ("## 4. A 包逐轮结果", ["accuracy", "macro_f1"]),
            "B": ("## 5. B 包逐轮结果", ["mae", "rmse"]),
            "C": ("## 6. C 包逐轮结果", ["mae", "nasa", "rmse"])}


def _num(text):
    text = text.strip().replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def parse_rounds(doc_text, header, metric_names):
    body = doc_text.split(header, 1)[1].split("\n## ", 1)[0]
    rows = []
    for line in body.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 1 + len(metric_names) + 1 or not cells[0].isdigit():
            continue
        row = {"round": int(cells[0])}
        for i, name in enumerate(metric_names):
            row[name] = _num(cells[1 + i])
        row["promoted"] = 1.0 if cells[1 + len(metric_names)] == "采用" else 0.0
        rest = cells[2 + len(metric_names):]
        if len(rest) >= 2:
            row["epochs"], row["lr"] = _num(rest[0]), _num(rest[1])  # “默认”解析为 None，不记录
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default="Annie")
    ap.add_argument("--project", default="zhixun-harness-official")
    args = ap.parse_args()
    for spec in RUNS:
        text = (DOCS / spec["doc"]).read_text(encoding="utf-8")
        config = {k: v for k, v in spec.items() if k not in ("scores", "name")}
        run = swanlab.init(project=args.project, workspace=args.workspace,
                           experiment_name=spec["name"], config=config)
        swanlab.log({f"score/{k}": v for k, v in spec["scores"].items()} | {"score/total": spec["total"]})
        if "54.84" in spec["name"]:
            for key, (header, names) in SECTIONS.items():
                for row in parse_rounds(text, header, names):
                    r = row.pop("round")
                    swanlab.log({f"{key}/{k}": v for k, v in row.items() if v is not None}, step=r)
        run.finish()
        print("synced", spec["name"])


if __name__ == "__main__":
    main()
