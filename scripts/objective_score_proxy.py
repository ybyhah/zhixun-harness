"""Build a local objective-score proxy from official baselines and round CSVs."""
import argparse
import csv
import json
import math
from pathlib import Path


FIELDS = (
    "profile", "metric", "baseline", "average_relative_gain",
    "worst_relative_gain", "recovery_rounds", "max_drawdown",
    "total_seconds", "promotions", "rejections", "adoption_rate",
    "final_best", "resource_efficiency",
)


def load_static_baselines(base_dir):
    result = {}
    for path in sorted(Path(base_dir).glob("cpu-*/training.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        profile, task = report.get("profile"), report.get("task")
        metric = "macro_f1" if task == "classification" else "rmse"
        value = report.get("validation_after", {}).get(metric)
        if profile not in ("A", "B", "C") or not isinstance(value, (int, float)):
            continue
        if profile in result or not math.isfinite(value):
            raise ValueError(f"invalid or duplicate static baseline: {profile}")
        result[profile] = {
            "metric": metric,
            "direction": "higher" if task == "classification" else "lower",
            "value": float(value),
        }
    if set(result) != {"A", "B", "C"}:
        raise ValueError("official A/B/C static baselines are required")
    return result


def summarize_rounds(profile, rows, baseline):
    values = [float(row["value"]) for row in rows]
    if not values:
        raise ValueError(f"no rounds for profile {profile}")
    scale = max(abs(baseline["value"]), 1e-12)
    if baseline["direction"] == "higher":
        gains = [(value - baseline["value"]) / scale for value in values]
        drawdowns, best = [], values[0]
        for value in values:
            best = max(best, value)
            drawdowns.append((best - value) / max(abs(best), 1e-12))
    else:
        gains = [(baseline["value"] - value) / scale for value in values]
        drawdowns, best = [], values[0]
        for value in values:
            best = min(best, value)
            drawdowns.append((value - best) / max(abs(best), 1e-12))
    worst_index = min(range(len(gains)), key=gains.__getitem__)
    recovery = 0 if gains[worst_index] >= 0 else next(
        (index - worst_index for index in range(worst_index + 1, len(gains))
         if gains[index] >= 0), None)
    seconds = sum(float(row["seconds"]) for row in rows)
    promotions = sum(int(row["accepted"]) for row in rows)
    average_gain = sum(gains) / len(gains)
    return {
        "profile": profile, "metric": baseline["metric"],
        "baseline": baseline["value"],
        "average_relative_gain": average_gain,
        "worst_relative_gain": min(gains), "recovery_rounds": recovery,
        "max_drawdown": max(drawdowns), "total_seconds": seconds,
        "promotions": promotions, "rejections": len(rows) - promotions,
        "adoption_rate": promotions / len(rows),
        "final_best": float(rows[-1]["best_value"]),
        "resource_efficiency": average_gain / seconds if seconds else 0.0,
    }


def write_reports(rows, report_dir):
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "objective_score_proxy.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    md_path = report_dir / "objective_score_proxy.md"
    lines = [
        "# 客观评分代理表", "",
        "本表基于官方包静态基线与本地逐轮结果计算，不等同于平台正式分数。", "",
        "| 包 | 指标 | 静态基线 | 平均相对提升 | 最差相对提升 | 恢复轮数 | 最大回撤 | 耗时秒 | 上线率 | 最终最优 |", 
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        recovery = "未恢复" if row["recovery_rounds"] is None else str(row["recovery_rounds"])
        lines.append(
            f"| {row['profile']} | {row['metric']} | {row['baseline']:.6f} | "
            f"{row['average_relative_gain']:.2%} | {row['worst_relative_gain']:.2%} | "
            f"{recovery} | {row['max_drawdown']:.2%} | {row['total_seconds']:.2f} | "
            f"{row['adoption_rate']:.1%} | {row['final_best']:.6f} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default="base")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--report-dir", default="reports/objective_score_proxy")
    args = parser.parse_args(argv)
    baselines = load_static_baselines(args.base_dir)
    rows = []
    for profile in ("A", "B", "C"):
        path = Path(args.outputs_dir) / f"official_{profile.lower()}_rounds.csv"
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows.append(summarize_rounds(
                profile, list(csv.DictReader(stream)), baselines[profile]))
    csv_path, md_path = write_reports(rows, args.report_dir)
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
