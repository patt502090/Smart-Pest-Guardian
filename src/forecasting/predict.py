"""Inference utilities for trained forecasting models."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import typer
from rich.console import Console

from .dataset import SequenceConfig, SequenceDataset, load_and_prepare
from .train_lstm import LSTMForecaster

console = Console()
app = typer.Typer(help="พยากรณ์การระบาดศัตรูพืชด้วยโมเดลที่เทรนแล้ว")


@app.command()
def predict_lstm(
    model_path: Path = typer.Argument(..., help="ไฟล์น้ำหนัก LSTM เช่น lstm_best.pt"),
    csv_path: Path = typer.Argument(..., help="ไฟล์ time series ล่าสุด"),
    output_path: Path = typer.Option(Path("results/forecasting/forecast.csv"), help="ไฟล์ผลลัพธ์"),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", help="อุปกรณ์"),
) -> None:
    """Load a trained LSTM model and forecast future pest counts."""
    if not model_path.exists():
        raise typer.BadParameter(f"ไม่พบไฟล์ {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    config_dict = checkpoint["config"]
    feature_columns = config_dict.get("feature_columns")
    group_columns = config_dict.get("group_columns")
    target_column = config_dict.get("target_column")
    seq_config = SequenceConfig(
        group_columns=group_columns,
        feature_columns=feature_columns,
        target_column=target_column,
        input_window=config_dict["input_window"],
        forecast_horizon=config_dict["forecast_horizon"],
    )
    df = load_and_prepare(csv_path, seq_config)
    dataset = SequenceDataset(df, seq_config)
    device_obj = torch.device(device)

    model = LSTMForecaster(
        num_features=config_dict["num_features"],
        hidden_size=config_dict["hidden_size"],
        num_layers=config_dict["num_layers"],
        horizon=config_dict["horizon"],
        dropout=config_dict["dropout"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device_obj)
    model.eval()

    predictions = []
    with torch.no_grad():
        for features, _, meta in dataset:
            features_tensor = torch.tensor(features, device=device_obj).unsqueeze(0)
            pred = model(features_tensor).cpu().numpy().flatten()
            group_key, forecast_time = meta
            predictions.append(
                {
                    "group": group_key if isinstance(group_key, (list, tuple)) else (group_key,),
                    "forecast_horizon": seq_config.forecast_horizon,
                    "target_column": target_column,
                    "forecast_end_date": forecast_time,
                    "predicted_values": pred.tolist(),
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_csv(output_path, index=False)
    console.log(f"บันทึกผลพยากรณ์ที่ {output_path}")


if __name__ == "__main__":
    app()
