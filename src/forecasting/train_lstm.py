"""Training script for LSTM-based pest outbreak forecasting."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import typer
from rich.console import Console

from .dataset import SequenceConfig, SequenceDataset, load_and_prepare

console = Console()
app = typer.Typer(help="เทรนโมเดล LSTM สำหรับพยากรณ์ศัตรูพืช")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TIMESERIES = PROJECT_ROOT / "data" / "processed" / "forecasting" / "pest_incidents_daily.csv"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "forecaster"


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


@app.command()
def train(
    csv_path: Path = typer.Option(DEFAULT_TIMESERIES, help="ไฟล์ time series"),
    target: str = typer.Option("count_total", help="คอลัมน์ตัวแปรเป้าหมาย"),
    group_columns: str = typer.Option("location_id", help="คอลัมน์กลุ่ม (คั่นด้วยจุลภาค)"),
    feature_columns: str = typer.Option("mean_confidence,temp_c,humidity", help="คอลัมน์คุณลักษณะ"),
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
) -> None:
    """Train an LSTM-based forecaster."""
    groups = [col.strip() for col in group_columns.split(",") if col.strip()]
    features = [col.strip() for col in feature_columns.split(",") if col.strip()]
    config = SequenceConfig(
        group_columns=groups,
        feature_columns=features,
        target_column=target,
        input_window=input_window,
        forecast_horizon=horizon,
    )
    console.rule("เทรน LSTM Forecaster")
    df = load_and_prepare(csv_path, config)
    dataset = SequenceDataset(df, config)
    train_set, val_set = train_val_split(dataset, val_ratio)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    device_obj = torch.device(device)
    model = LSTMForecaster(len(features), hidden_size, num_layers, horizon, dropout).to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    best_rmse = float("inf")
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "lstm_best.pt"

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device_obj)
        val_loss, val_rmse = evaluate(model, val_loader, criterion, device_obj)
        console.log(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_RMSE={val_rmse:.4f}"
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
                },
                weights_path,
            )
            console.log(f"[green]บันทึกโมเดลใหม่ที่ {weights_path} (RMSE={best_rmse:.4f})[/green]")

    console.log("การเทรนเสร็จสมบูรณ์")


if __name__ == "__main__":
    app()
