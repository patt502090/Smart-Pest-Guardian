"""End-to-end orchestration script for Smart Pest Guardian."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console

console = Console()
app = typer.Typer(help="รัน workflow หลักแบบครบวงจร")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


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
        PROJECT_ROOT / "data" / "processed" / "detection" / "pests_2xlvx_yolo" / "dataset.yaml",
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
    output_dir: Path = typer.Option(Path("results/detections"), help="โฟลเดอร์สำหรับผลการตรวจจับ"),
    location_id: str = typer.Option("site-1", help="รหัสพื้นที่เพื่อส่งต่อสู่ time-series"),
    captured_at: str | None = typer.Option(
        None, help="กำหนดเวลาเก็บภาพแบบ ISO8601 ถ้าต้องการใช้ค่าเดียวกันทุกภาพหรือจุดเริ่ม"
    ),
    timestamp_mode: str = typer.Option(
        "file",
        help="วิธีกำหนดเวลา: file=ตามไฟล์, fixed=ใช้ captured-at, sequence=เวลาดีดต่อเนื่อง",
    ),
    sequence_step_minutes: int = typer.Option(1440, help="ช่วงเวลาห่างกันในโหมด sequence (นาที)"),
) -> None:
    """Run detection on a folder of images."""
    command = [
        PYTHON,
        str(PROJECT_ROOT / "src" / "detection" / "predict.py"),
        "run-images",
        str(weights),
        str(images),
        "--output-dir",
        str(output_dir),
        "--location-id",
        location_id,
    ]
    if captured_at:
        command.extend(["--captured-at", captured_at])
    if timestamp_mode:
        command.extend(["--timestamp-mode", timestamp_mode])
    if timestamp_mode.lower() == "sequence":
        command.extend(["--sequence-step-minutes", str(sequence_step_minutes)])
    _run(command)


@app.command()
def detect_to_forecast(
    weights: Path = typer.Argument(..., help="ไฟล์น้ำหนัก best.pt"),
    images: Path = typer.Argument(..., help="โฟลเดอร์ภาพที่ต้องการตรวจจับ"),
    location_id: str = typer.Option("site-1", help="รหัสพื้นที่"),
    captured_at: str | None = typer.Option(
        None, help="กำหนดเวลาเก็บภาพแบบ ISO8601 หากต้องการใช้ค่าเดียว"
    ),
    timestamp_mode: str = typer.Option(
        "file",
        help="วิธีกำหนดเวลา: file=ตามไฟล์, fixed=ใช้ captured-at, sequence=เวลาดีดต่อเนื่อง",
    ),
    sequence_step_minutes: int = typer.Option(1440, help="ช่วงเวลาห่างกันในโหมด sequence (นาที)"),
    detections_dir: Path = typer.Option(Path("results/detections"), help="ตำแหน่งบันทึก detection"),
    aggregated_csv: Path = typer.Option(
        PROJECT_ROOT / "data" / "processed" / "forecasting" / "pest_incidents_daily.csv",
        help="ไฟล์ time series ปลายทาง",
    ),
    run_forecaster: bool = typer.Option(True, help="เทรนโมเดลพยากรณ์หลัง aggregate"),
    feature_columns: str = typer.Option(
        "count_total,mean_confidence",
        help="คอลัมน์ feature สำหรับ forecaster (คั่นด้วยจุลภาค)",
    ),
) -> None:
    """Detect pests, aggregate to time series, and optionally train forecaster."""
    detect_cmd = [
        PYTHON,
        str(PROJECT_ROOT / "src" / "detection" / "predict.py"),
        "run-images",
        str(weights),
        str(images),
        "--output-dir",
        str(detections_dir),
        "--location-id",
        location_id,
    ]
    if captured_at:
        detect_cmd.extend(["--captured-at", captured_at])
    if timestamp_mode:
        detect_cmd.extend(["--timestamp-mode", timestamp_mode])
    if timestamp_mode.lower() == "sequence":
        detect_cmd.extend(["--sequence-step-minutes", str(sequence_step_minutes)])
    _run(detect_cmd)

    detections_csv = detections_dir / "detections.csv"
    aggregate_cmd = [
        PYTHON,
        str(PROJECT_ROOT / "scripts" / "prepare_detection_dataset.py"),
        "aggregate",
        str(detections_csv),
        "--output-csv",
        str(aggregated_csv),
    ]
    _run(aggregate_cmd)

    if run_forecaster:
        features = [col.strip() for col in feature_columns.split(",") if col.strip()]
        feature_arg = ",".join(features) if features else "count_total,mean_confidence"
        forecaster_cmd = [
            PYTHON,
            str(PROJECT_ROOT / "src" / "forecasting" / "train_lstm.py"),
            "train",
            "--csv-path",
            str(aggregated_csv),
            "--feature-columns",
            feature_arg,
        ]
        _run(forecaster_cmd)


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
