"""Optional trainer using Temporal Fusion Transformer via pytorch-forecasting."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import typer
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from rich.console import Console

console = Console()
app = typer.Typer(help="เทรน Temporal Fusion Transformer")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TIMESERIES = PROJECT_ROOT / "data" / "processed" / "forecasting" / "pest_incidents_daily.csv"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "forecaster"


@app.command()
def train(
    csv_path: Path = typer.Option(DEFAULT_TIMESERIES, help="ไฟล์ time series"),
    time_idx: str = typer.Option("time_idx", help="ชื่อคอลัมน์ลำดับเวลา"),
    target: str = typer.Option("count_total", help="คอลัมน์เป้าหมาย"),
    group_columns: str = typer.Option("location_id", help="กลุ่ม"),
    max_encoder_length: int = typer.Option(30, help="จำนวนจุดข้อมูลย้อนหลัง"),
    max_prediction_length: int = typer.Option(7, help="จำนวนวันที่พยากรณ์"),
    batch_size: int = typer.Option(64, help="ขนาด batch"),
    epochs: int = typer.Option(30, help="จำนวน epoch"),
    gpus: int = typer.Option(1 if pl.utilities.device_parser.num_cuda_devices() else 0, help="จำนวน GPU"),
    accelerator: str = typer.Option("auto", help="accelerator"),
    output_dir: Path = typer.Option(DEFAULT_MODEL_DIR, help="ตำแหน่งบันทึกโมเดล"),
) -> None:
    """Train a TFT model (requires pytorch-forecasting)."""
    df = pd.read_csv(csv_path)
    if time_idx not in df.columns:
        df = df.sort_values("date")
        df[time_idx] = range(len(df))

    group_cols = [col.strip() for col in group_columns.split(",") if col.strip()]
    training_cutoff = df[time_idx].max() - max_prediction_length

    training = TimeSeriesDataSet(
        df[df[time_idx] <= training_cutoff],
        time_idx=time_idx,
        target=target,
        group_ids=group_cols,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=[time_idx],
        time_varying_unknown_reals=[target],
        target_normalizer=GroupNormalizer(groups=group_cols, transformation="softplus"),
    )

    validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
    train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=1e-3,
        hidden_size=32,
        attention_head_size=4,
        dropout=0.1,
        loss=QuantileLoss(),
        output_size=7,
        reduce_on_plateau_patience=3,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=gpus if gpus > 0 else 1,
        gradient_clip_val=0.1,
        enable_checkpointing=True,
        default_root_dir=str(output_dir),
        log_every_n_steps=5,
    )

    trainer.fit(tft, train_loader, val_loader)
    console.log("การเทรน TFT เสร็จสมบูรณ์")


if __name__ == "__main__":
    app()
