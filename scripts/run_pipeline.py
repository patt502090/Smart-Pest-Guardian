"""End-to-end orchestration script for Smart Pest Guardian."""
from __future__ import annotations

import subprocess
from pathlib import Path

import typer
from rich.console import Console

console = Console()
app = typer.Typer(help="รัน workflow หลักแบบครบวงจร")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = "python"


def _run(command: list[str]) -> None:
    console.log("รันคำสั่ง: " + " ".join(command))
    subprocess.run(command, check=True)


@app.command()
def download_climate(
    lat: float = typer.Option(13.75, help="ละติจูดพื้นที่"),
    lon: float = typer.Option(100.5, help="ลองจิจูด"),
    start: str = typer.Option("20240101", help="วันเริ่ม"),
    end: str = typer.Option("20241231", help="วันจบ"),
) -> None:
    """Download climate data from NASA POWER."""
    _run([
        PYTHON,
        str(PROJECT_ROOT / "src" / "data" / "download.py"),
        "download-nasa-power",
        "--latitude",
        str(lat),
        "--longitude",
        str(lon),
        "--start",
        start,
        "--end",
        end,
    ])


@app.command()
def train_detector(
    data_yaml: Path = typer.Option(
        PROJECT_ROOT / "data" / "processed" / "detection" / "ai_challenger_yolo" / "dataset.yaml",
        help="ไฟล์ dataset.yaml",
    ),
    model: str = typer.Option("yolov8n.pt", help="ไฟล์ weights ตั้งต้น"),
) -> None:
    """Train YOLO detector."""
    _run([
        PYTHON,
        str(PROJECT_ROOT / "src" / "detection" / "train.py"),
        "train",
        "--data-config",
        str(data_yaml),
        "--model",
        model,
    ])


@app.command()
def run_detection(
    weights: Path = typer.Argument(..., help="ไฟล์น้ำหนัก best.pt"),
    images: Path = typer.Argument(..., help="โฟลเดอร์ภาพ"),
) -> None:
    """Run detection on a folder of images."""
    _run([
        PYTHON,
        str(PROJECT_ROOT / "src" / "detection" / "predict.py"),
        "run-images",
        str(weights),
        str(images),
    ])


@app.command()
def train_forecaster(
    csv_path: Path = typer.Option(
        PROJECT_ROOT / "data" / "processed" / "forecasting" / "pest_incidents_daily.csv",
        help="ไฟล์ time series",
    )
) -> None:
    """Train LSTM forecaster."""
    _run([
        PYTHON,
        str(PROJECT_ROOT / "src" / "forecasting" / "train_lstm.py"),
        "train",
        "--csv-path",
        str(csv_path),
    ])


@app.command()
def build_report() -> None:
    """Placeholder for automated report generation."""
    console.print("ฟังก์ชันนี้จะถูกรันเมื่อเตรียม template รายงาน (กำลังพัฒนา)")


if __name__ == "__main__":
    app()
