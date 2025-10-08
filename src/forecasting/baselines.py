"""Baseline forecaster evaluation for comparison with LSTM model."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import typer
from rich.console import Console

import matplotlib.pyplot as plt

from src.forecasting.dataset import SequenceConfig, load_and_prepare

console = Console()

DEFAULT_CSV = Path("data/processed/forecasting/pest_incidents_daily_integration.csv")
DEFAULT_REPORT_DIR = Path("reports/metrics/forecaster")


def build_group_label(group_columns: List[str], group_key) -> str:
    if not group_columns:
        return "overall"
    if len(group_columns) == 1:
        return f"{group_columns[0]}={group_key}"
    if isinstance(group_key, tuple):
        return ", ".join(f"{col}={val}" for col, val in zip(group_columns, group_key))
    return ", ".join(f"{col}={val}" for col, val in zip(group_columns, (group_key,)))


def compute_metrics(preds: np.ndarray, actuals: np.ndarray) -> Dict[str, float]:
    diff = preds - actuals
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    mask = np.abs(actuals) >= 1e-3
    mape = float(np.mean(np.abs(diff[mask] / actuals[mask])) * 100) if np.any(mask) else float("nan")
    return {"rmse": rmse, "mae": mae, "mape": mape}


def evaluate_baselines(
    series: np.ndarray,
    input_window: int,
    horizon: int,
) -> Dict[str, Dict[str, float]]:
    num_windows = len(series) - input_window - horizon + 1
    if num_windows <= 0:
        raise ValueError("ข้อมูลไม่เพียงพอสำหรับประเมิน baseline")

    naive_preds: List[np.ndarray] = []
    ma_preds: List[np.ndarray] = []
    actual_windows: List[np.ndarray] = []

    for start in range(num_windows):
        history = series[start : start + input_window]
        future = series[start + input_window : start + input_window + horizon]
        if history.size < input_window or future.size < horizon:
            continue
        actual_windows.append(future)

        last_value = history[-1]
        naive_preds.append(np.full(horizon, last_value))
        moving_avg = history.mean()
        ma_preds.append(np.full(horizon, moving_avg))

    if not actual_windows:
        raise ValueError("ข้อมูลไม่เพียงพอสำหรับประเมิน baseline")

    actual_arr = np.concatenate(actual_windows)
    naive_arr = np.concatenate(naive_preds)
    ma_arr = np.concatenate(ma_preds)

    return {
        "naive_last": compute_metrics(naive_arr, actual_arr),
        "moving_average": compute_metrics(ma_arr, actual_arr),
    }


def find_feasible_window(series_length: int, desired_window: int, desired_horizon: int) -> Tuple[int, int]:
    """Reduce window/horizon gracefully until at leastหนึ่งหน้าต่างใช้งานได้."""

    if series_length < 2:
        raise ValueError("ข้อมูลมีจำนวนน้อยเกินไปสำหรับ baseline")

    max_window = min(desired_window, series_length - 1)
    for window in range(max_window, 0, -1):
        max_horizon = min(desired_horizon, series_length - window)
        for horizon in range(max_horizon, 0, -1):
            if series_length - window - horizon + 1 > 0:
                return window, horizon
    raise ValueError("ข้อมูลไม่เพียงพอ แม้ลด window/horizon แล้ว")


def compare(
    csv_path: Path = typer.Option(DEFAULT_CSV, help="ไฟล์ข้อมูลรวมรายวัน"),
    target: str = typer.Option("count_total", help="คอลัมน์ตัวแปรที่ต้องการพยากรณ์"),
    group_columns: str = typer.Option("location_id", help="คอลัมน์กลุ่ม (คั่นด้วยจุลภาค)"),
    input_window: int = typer.Option(14, help="จำนวนวันย้อนหลังที่ใช้ประเมิน"),
    horizon: int = typer.Option(7, help="จำนวนวันล่วงหน้าที่เปรียบเทียบ"),
    report_dir: Path = typer.Option(DEFAULT_REPORT_DIR, help="โฟลเดอร์บันทึกผล"),
) -> None:
    groups = [col.strip() for col in group_columns.split(",") if col.strip()]
    config = SequenceConfig(
        group_columns=groups,
        feature_columns=[target],
        target_column=target,
    )
    df = load_and_prepare(csv_path, config, add_time_features=False)
    report_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows: List[Dict[str, object]] = []

    iter_groups: Iterable[Tuple] = (
        df.groupby(groups) if groups else [("overall", df)]
    )

    adjustments: List[Dict[str, object]] = []
    for group_key, group_df in iter_groups:
        series = group_df[target].to_numpy(dtype=np.float32)
        if series.ndim > 1:
            console.log(
                f"[blue]ตรวจพบคอลัมน์ซ้ำในกลุ่ม {group_key} ใช้คอลัมน์สุดท้ายเป็น target[/blue]"
            )
            series = series[:, -1]
        try:
            effective_window, effective_horizon = find_feasible_window(len(series), input_window, horizon)
            if effective_window != input_window or effective_horizon != horizon:
                adjustments.append({
                    "group": build_group_label(groups, group_key if groups else None),
                    "input_window": effective_window,
                    "horizon": effective_horizon,
                })
                console.log(
                    f"[cyan]ปรับกลุ่ม {group_key} -> window={effective_window}, horizon={effective_horizon}[/cyan]"
                )

            metrics = evaluate_baselines(series, effective_window, effective_horizon)
        except ValueError as exc:
            console.log(f"[yellow]ข้ามกลุ่ม {group_key}: {exc}[/yellow]")
            continue

        label = build_group_label(groups, group_key if groups else None)
        for name, metric in metrics.items():
            rows.append({
                "group": label,
                "baseline": name,
                "rmse": metric["rmse"],
                "mae": metric["mae"],
                "mape": metric["mape"],
                "input_window": effective_window,
                "horizon": effective_horizon,
            })

    if not rows:
        raise typer.Exit("ไม่สามารถประเมิน baseline ได้เนื่องจากข้อมูลไม่พอ")

    result_df = pd.DataFrame(rows)
    output_path = report_dir / f"baseline_metrics_{run_id}.csv"
    result_df.to_csv(output_path, index=False)

    # Attempt to summarize latest LSTM validation metrics for comparison
    lstm_metrics = None
    history_files = sorted(report_dir.glob("lstm_*_history.csv"))
    if history_files:
        latest_history = history_files[-1]
        history_df = pd.read_csv(latest_history)
        best_row = history_df.loc[history_df["val_rmse"].idxmin()]
        lstm_metrics = {
            "rmse": float(best_row["val_rmse"]),
            "mae": float(best_row.get("val_mae", np.nan)),
            "mape": float(best_row.get("val_mape", np.nan)),
            "source": latest_history.name,
        }

    plot_path = report_dir / f"baseline_comparison_{run_id}.png"
    create_comparison_plot(result_df, lstm_metrics, plot_path)

    console.rule("Baseline Comparison")
    console.log(result_df)
    console.log(f"[green]บันทึกผลที่ {output_path}[/green]")
    console.log(f"[green]บันทึกกราฟที่ {plot_path}[/green]")

    if adjustments:
        console.rule("การปรับพารามิเตอร์ baseline")
        console.log(pd.DataFrame(adjustments))


def create_comparison_plot(baseline_df: pd.DataFrame, lstm_metrics: Dict[str, float] | None, output_path: Path) -> None:
    pivot = baseline_df.groupby("baseline")[['rmse', 'mae', 'mape']].mean()
    if lstm_metrics:
        pivot.loc["lstm_model"] = [lstm_metrics["rmse"], lstm_metrics["mae"], lstm_metrics["mape"]]

    metrics = ["rmse", "mae", "mape"]
    x = np.arange(len(pivot.index))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, metric in enumerate(metrics):
        if metric not in pivot.columns:
            continue
        offset = (idx - (len(metrics) - 1) / 2) * width
        ax.bar(
            x + offset,
            pivot[metric].to_numpy(),
            width=width,
            label=metric.upper(),
        )

    ax.set_xticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.index, rotation=0)
    ax.set_ylabel("Error Value")
    ax.set_title("LSTM vs Baseline Metrics")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    typer.run(compare)
