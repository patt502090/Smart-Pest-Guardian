"""Hyperparameter sweep utility for the LSTM forecaster."""
from __future__ import annotations

from datetime import datetime
from itertools import product
from pathlib import Path
from typing import List

import pandas as pd
import torch
import typer
from rich.console import Console

from src.forecasting.train_lstm import train as train_lstm

console = Console()
app = typer.Typer(help="รัน hyperparameter sweep สำหรับ LSTM forecaster")

DEFAULT_CSV = Path("data/processed/forecasting/pest_incidents_daily_integration.csv")
DEFAULT_REPORT_DIR = Path("reports/metrics/forecaster")
DEFAULT_MODEL_DIR = Path("models/forecaster/sweeps")


def _parse_list(value: str, cast) -> List:
    if not value:
        return []
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@app.command()
def sweep(
    csv_path: Path = typer.Option(DEFAULT_CSV, help="ไฟล์ time series ที่ใช้เทรน"),
    group_columns: str = typer.Option("location_id", help="คอลัมน์กลุ่ม (คั่นด้วยจุลภาค)"),
    feature_columns: str = typer.Option("mean_confidence,count_total", help="คอลัมน์ฟีเจอร์"),
    target: str = typer.Option("count_total", help="คอลัมน์เป้าหมาย"),
    input_window: int = typer.Option(14, help="จำนวนวันย้อนหลัง"),
    horizon: int = typer.Option(7, help="จำนวนวันล่วงหน้า"),
    epochs: int = typer.Option(30, help="จำนวน epoch ต่อการทดลอง"),
    batch_size: int = typer.Option(32, help="ขนาด batch"),
    num_layers: int = typer.Option(2, help="จำนวนชั้น LSTM"),
    hidden_sizes: str = typer.Option("96,128", help="ชุดค่า hidden size (คั่นด้วยจุลภาค)"),
    learning_rates: str = typer.Option("1e-3,5e-4", help="ชุดค่า learning rate"),
    dropouts: str = typer.Option("0.2", help="ชุดค่า dropout"),
    val_ratio: float = typer.Option(0.2, help="สัดส่วน validation"),
    device: str = typer.Option(DEFAULT_DEVICE, help="อุปกรณ์ที่ใช้ (cuda หรือ cpu)"),
    report_dir: Path = typer.Option(DEFAULT_REPORT_DIR / "sweeps", help="โฟลเดอร์บันทึกผลการทดลอง"),
    output_dir: Path = typer.Option(DEFAULT_MODEL_DIR, help="โฟลเดอร์เก็บน้ำหนักโมเดล"),
) -> None:
    hidden_list = _parse_list(hidden_sizes, int)
    lr_list = _parse_list(learning_rates, float)
    dropout_list = _parse_list(dropouts, float)

    if not hidden_list or not lr_list or not dropout_list:
        raise typer.Exit("โปรดระบุ hidden_sizes, learning_rates และ dropouts อย่างน้อย 1 ค่า")

    groups = [col.strip() for col in group_columns.split(",") if col.strip()]
    features = [col.strip() for col in feature_columns.split(",") if col.strip()]
    report_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.rule("เริ่ม Hyperparameter Sweep")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    records = []

    for hidden_size, lr, dropout in product(hidden_list, lr_list, dropout_list):
        run_label = f"hs{hidden_size}_lr{lr}_do{dropout}"
        console.log(f"[cyan]ทดลอง {run_label}[/cyan]")
        run_output = output_dir / run_label
        run_output.mkdir(parents=True, exist_ok=True)

        summary = train_lstm(
            csv_path=csv_path,
            target=target,
            group_columns=",".join(groups),
            feature_columns=",".join(features),
            input_window=input_window,
            horizon=horizon,
            epochs=epochs,
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=lr,
            val_ratio=val_ratio,
            device=device,
            output_dir=run_output,
            report_dir=report_dir,
        )

        best = summary["best_metrics"]
        records.append(
            {
                "run_label": run_label,
                "hidden_size": hidden_size,
                "learning_rate": lr,
                "dropout": dropout,
                "best_rmse": best.get("rmse"),
                "best_mae": best.get("mae"),
                "best_mape": best.get("mape"),
                "history_csv": Path(summary["history_csv"]).as_posix(),
                "history_plot": Path(summary["history_plot"]).as_posix(),
                "weights_path": Path(summary["weights_path"]).as_posix(),
            }
        )

    sweep_df = pd.DataFrame(records)
    output_path = report_dir / f"lstm_sweep_{timestamp}.csv"
    sweep_df.to_csv(output_path, index=False)
    console.rule("สรุปผล Hyperparameter Sweep")
    console.log(sweep_df)
    console.log(f"[green]บันทึกผลไว้ที่ {output_path}[/green]")


if __name__ == "__main__":
    typer.run(sweep)
