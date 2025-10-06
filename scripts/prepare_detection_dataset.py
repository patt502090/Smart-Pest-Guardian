"""CLI for preparing datasets for detection and forecasting."""
from __future__ import annotations

from pathlib import Path

import typer

from src.data import preprocess

app = typer.Typer(help="เตรียมข้อมูลสำหรับงานตรวจจับและพยากรณ์")


@app.command()
def convert_ai_challenger(
    source_dir: Path = typer.Option(Path("data/raw/AI_Challenger_Pest")),
    output_dir: Path = typer.Option(Path("data/processed/detection/ai_challenger_yolo")),
) -> None:
    preprocess.convert_ai_challenger_to_yolo(source_dir=source_dir, output_dir=output_dir)


@app.command()
def aggregate(
    detections_csv: Path = typer.Argument(Path("results/detections/detections.csv")),
    output_csv: Path = typer.Option(Path("data/processed/forecasting/pest_incidents_daily.csv")),
) -> None:
    preprocess.aggregate_detection_to_timeseries(detections_csv, output_csv)


if __name__ == "__main__":
    app()
