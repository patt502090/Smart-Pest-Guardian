"""Utilities to consolidate experiment metrics into markdown reports."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import typer
from rich.console import Console

from .metrics import compute_forecasting_metrics

console = Console()
app = typer.Typer(help="สร้างรายงานสรุปผลการทดลอง")


@app.command()
def summarize(
    detection_metrics_path: Path = typer.Option(Path("models/detector/metrics.csv"), help="ไฟล์ metric YOLO"),
    forecast_truth: Path = typer.Option(Path("data/processed/forecasting/eval_truth.csv"), help="ค่าจริง"),
    forecast_pred: Path = typer.Option(Path("results/forecasting/forecast.csv"), help="ค่าพยากรณ์"),
    output_markdown: Path = typer.Option(Path("reports/drafts/experiment_summary.md"), help="ไฟล์สรุป"),
    target_column: str = typer.Option("count_total", help="คอลัมน์เป้าหมาย"),
) -> None:
    """Generate a markdown summary combining detection and forecasting metrics."""
    detection_section = "ไม่พบไฟล์ metric ของการตรวจจับ"
    if detection_metrics_path.exists():
        det_df = pd.read_csv(detection_metrics_path)
        latest = det_df.iloc[-1]
        detection_section = (
            f"- mAP50: **{latest.get('metrics/mAP50(B)', 'N/A')}**\n"
            f"- Precision: **{latest.get('metrics/precision(B)', 'N/A')}**\n"
            f"- Recall: **{latest.get('metrics/recall(B)', 'N/A')}**"
        )

    forecast_section = "ไม่พบข้อมูลพยากรณ์"
    if forecast_truth.exists() and forecast_pred.exists():
        truth_df = pd.read_csv(forecast_truth)
        pred_df = pd.read_csv(forecast_pred)
        merged = truth_df.merge(pred_df, on=["date", "location_id"], suffixes=("_true", "_pred"))
        metrics = compute_forecasting_metrics(
            merged[f"{target_column}_true"].values,
            merged[f"{target_column}_pred"].values,
        )
        forecast_section = (
            f"- MAE: **{metrics['mae']:.3f}**\n"
            f"- RMSE: **{metrics['rmse']:.3f}**\n"
            f"- MAPE: **{metrics['mape']:.2f}%**"
        )

    report = f"""# สรุปผลการทดลองล่าสุด

## การตรวจจับศัตรูพืช
{detection_section}

## การพยากรณ์การระบาด
{forecast_section}

"""
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(report, encoding="utf-8")
    console.log(f"สร้างรายงานสรุปที่ {output_markdown}")


if __name__ == "__main__":
    app()
