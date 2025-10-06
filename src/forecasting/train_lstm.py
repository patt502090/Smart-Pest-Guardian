"""Training script for LSTM-based pest outbreak forecasting."""
from __future__ import annotations

import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import typer
from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.forecasting.dataset import SequenceConfig, SequenceDataset, load_and_prepare

console = Console()
app = typer.Typer(help="เทรนโมเดล LSTM สำหรับพยากรณ์ศัตรูพืช")
DEFAULT_TIMESERIES = PROJECT_ROOT / "data" / "processed" / "forecasting" / "pest_incidents_daily.csv"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "forecaster"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "metrics" / "forecaster"


def slugify(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return value or "overall"


def build_group_label(group_columns: List[str], group_key) -> str:
    if not group_columns:
        return "overall"
    if len(group_columns) == 1:
        return f"{group_columns[0]}={group_key}"
    key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
    return ", ".join(f"{col}={val}" for col, val in zip(group_columns, key_tuple))


def select_group_frame(df: pd.DataFrame, group_columns: List[str], group_key):
    if not group_columns:
        return df
    if len(group_columns) == 1:
        return df[df[group_columns[0]] == group_key]
    key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
    mask = pd.Series(True, index=df.index)
    for column, value in zip(group_columns, key_tuple):
        mask &= df[column] == value
    return df[mask]


def export_training_history(report_dir: Path, run_name: str, history_df: pd.DataFrame) -> Dict[str, Path]:
    csv_path = report_dir / f"{run_name}_history.csv"
    json_path = report_dir / f"{run_name}_history.json"
    history_df.to_csv(csv_path, index=False)
    history_df.to_json(json_path, orient="records", indent=2)
    return {"csv": csv_path, "json": json_path}


def plot_training_curves(report_dir: Path, run_name: str, history_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history_df["epoch"], history_df["val_rmse"], label="val_rmse", color="orange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RMSE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = report_dir / f"{run_name}_training.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def serialize_scalers(dataset: SequenceDataset) -> List[Dict[str, List[float]]]:
    serialized = []
    for key, scaler in dataset.group_scalers.items():
        serialized.append(
            {
                "key": [*key],
                "mean": scaler["mean"].tolist(),
                "std": scaler["std"].tolist(),
            }
        )
    return serialized


def plot_forecast_outputs(
    model: "LSTMForecaster",
    dataset: SequenceDataset,
    df: pd.DataFrame,
    config: SequenceConfig,
    device: torch.device,
    report_dir: Path,
    run_name: str,
) -> List[Path]:
    model.eval()
    horizon = config.forecast_horizon
    generated_paths: List[Path] = []
    last_indices: Dict = {}
    for idx, (group_key, _) in enumerate(dataset.groups):
        last_indices[group_key] = idx

    for group_key, idx in last_indices.items():
        inputs = torch.tensor(dataset.data[idx], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            preds_norm = model(inputs).cpu().numpy().flatten()
        scaler = dataset.get_scaler(group_key)
        target_mean = scaler["mean"][-1]
        target_std = scaler["std"][-1]
        preds = preds_norm * target_std + target_mean
        actual_norm = dataset.targets[idx]
        actual = actual_norm * target_std + target_mean

        group_df = select_group_frame(df, config.group_columns or [], group_key)
        if len(group_df) < horizon:
            continue
        tail_df = group_df.tail(horizon)
        dates = tail_df[config.time_column]
        label = build_group_label(config.group_columns or [], group_key)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(dates, actual, label="Actual", marker="o")
        ax.plot(dates, preds, label="Forecast", marker="x")
        ax.set_title(f"Forecast vs Actual ({label})")
        ax.set_ylabel(config.target_column)
        ax.set_xlabel("Date")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()

        out_path = report_dir / f"{run_name}_forecast_{slugify(label)}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        generated_paths.append(out_path)

    return generated_paths


def train_val_split(dataset: SequenceDataset, val_ratio: float = 0.2) -> Tuple[Dataset, Dataset]:
    total = len(dataset)
    val_size = max(1, int(total * val_ratio))
    indices = np.arange(total)
    train_idx = indices[:-val_size]
    val_idx = indices[-val_size:]

    class Subset(Dataset):
        def __init__(self, base: SequenceDataset, idx: np.ndarray) -> None:
            self.base = base
            self.idx = idx

        def __len__(self) -> int:
            return len(self.idx)

        def __getitem__(self, index: int):
            base_idx = self.idx[index]
            features, target, meta = self.base[base_idx]
            return torch.tensor(features), torch.tensor(target), meta

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


class LSTMForecaster(nn.Module):
    def __init__(self, num_features: int, hidden_size: int, num_layers: int, horizon: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        prediction = self.projection(last_hidden)
        return prediction


def train_epoch(model, dataloader, criterion, optimizer, device: torch.device) -> float:
    model.train()
    epoch_loss = 0.0
    for features, targets, _ in dataloader:
        features = features.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        preds = model(features)
        loss = criterion(preds, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item() * features.size(0)
    return epoch_loss / len(dataloader.dataset)


def evaluate(model, dataloader, criterion, device: torch.device) -> Tuple[float, float]:
    model.eval()
    epoch_loss = 0.0
    mse_sum = 0.0
    with torch.no_grad():
        for features, targets, _ in dataloader:
            features = features.to(device)
            targets = targets.to(device)
            preds = model(features)
            loss = criterion(preds, targets)
            epoch_loss += loss.item() * features.size(0)
            mse_sum += torch.mean((preds - targets) ** 2).item() * features.size(0)
    rmse = np.sqrt(mse_sum / len(dataloader.dataset))
    return epoch_loss / len(dataloader.dataset), rmse


def collate_batch(batch):
    features, targets, metas = zip(*batch)
    return torch.stack(features), torch.stack(targets), list(metas)


@app.command()
def train(
    csv_path: Path = typer.Option(DEFAULT_TIMESERIES, help="ไฟล์ time series"),
    target: str = typer.Option("count_total", help="คอลัมน์ตัวแปรเป้าหมาย"),
    group_columns: str = typer.Option("location_id", help="คอลัมน์กลุ่ม (คั่นด้วยจุลภาค)"),
    feature_columns: str = typer.Option("mean_confidence,count_total", help="คอลัมน์คุณลักษณะ"),
    input_window: int = typer.Option(14, help="จำนวนวันย้อนหลังที่ใช้เป็น input"),
    horizon: int = typer.Option(7, help="จำนวนวันล่วงหน้าที่พยากรณ์"),
    epochs: int = typer.Option(50, help="จำนวน epoch"),
    batch_size: int = typer.Option(32, help="ขนาด batch"),
    hidden_size: int = typer.Option(128, help="ขนาด hidden state"),
    num_layers: int = typer.Option(2, help="จำนวนชั้น LSTM"),
    dropout: float = typer.Option(0.2, help="อัตรา dropout"),
    learning_rate: float = typer.Option(1e-3, help="อัตราเรียนรู้"),
    val_ratio: float = typer.Option(0.2, help="สัดส่วน validation"),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", help="อุปกรณ์"),
    output_dir: Path = typer.Option(DEFAULT_MODEL_DIR, help="โฟลเดอร์บันทึกโมเดล"),
    report_dir: Path = typer.Option(DEFAULT_REPORT_DIR, help="โฟลเดอร์บันทึกกราฟและสรุป"),
) -> None:
    """Train an LSTM-based forecaster."""
    groups = [col.strip() for col in group_columns.split(",") if col.strip()]
    features = [col.strip() for col in feature_columns.split(",") if col.strip()]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"lstm_{run_id}"
    config = SequenceConfig(
        group_columns=groups,
        feature_columns=features,
        target_column=target,
        input_window=input_window,
        forecast_horizon=horizon,
    )
    console.rule("เทรน LSTM Forecaster")
    df = load_and_prepare(csv_path, config)
    report_dir.mkdir(parents=True, exist_ok=True)
    dataset = SequenceDataset(df, config)
    train_set, val_set = train_val_split(dataset, val_ratio)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_set, batch_size=batch_size, collate_fn=collate_batch)

    device_obj = torch.device(device)
    model = LSTMForecaster(len(features), hidden_size, num_layers, horizon, dropout).to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    history_records: List[Dict[str, float]] = []

    best_rmse = float("inf")
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "lstm_best.pt"

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device_obj)
        val_loss, val_rmse = evaluate(model, val_loader, criterion, device_obj)
        console.log(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_RMSE={val_rmse:.4f}"
        )
        history_records.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_rmse": float(val_rmse),
            }
        )
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "num_features": len(features),
                        "hidden_size": hidden_size,
                        "num_layers": num_layers,
                        "horizon": horizon,
                        "dropout": dropout,
                        "feature_columns": features,
                        "group_columns": groups,
                        "target_column": target,
                        **asdict(config),
                    },
                    "history": history_records.copy(),
                    "best_epoch": epoch,
                    "best_rmse": float(best_rmse),
                    "scalers": serialize_scalers(dataset),
                    "run_name": run_name,
                    "timestamp": run_id,
                },
                weights_path,
            )
            console.log(f"[green]บันทึกโมเดลใหม่ที่ {weights_path} (RMSE={best_rmse:.4f})[/green]")

    history_df = pd.DataFrame(history_records)
    artifacts = export_training_history(report_dir, run_name, history_df)
    history_plot_path = plot_training_curves(report_dir, run_name, history_df)

    checkpoint = torch.load(weights_path, map_location=device_obj)
    model.load_state_dict(checkpoint["model_state"])
    forecast_paths = plot_forecast_outputs(model, dataset, df, config, device_obj, report_dir, run_name)

    console.log(f"บันทึกประวัติการเทรน (CSV): {artifacts['csv']}")
    console.log(f"บันทึกประวัติการเทรน (JSON): {artifacts['json']}")
    console.log(f"บันทึกกราฟ loss/RMSE: {history_plot_path}")
    for path in forecast_paths:
        console.log(f"[green]สร้างกราฟ forecast: {path}[/green]")
    console.log(f"การเทรนเสร็จสมบูรณ์ (best RMSE={best_rmse:.4f})")


if __name__ == "__main__":
    app()
