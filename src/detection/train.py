"""Training utilities for YOLOv8-based pest detection."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from ultralytics import YOLO

console = Console()
app = typer.Typer(help="เทรนโมเดลตรวจจับศัตรูพืชด้วย YOLOv8")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "processed" / "detection" / "ai_challenger_yolo" / "dataset.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "detector"


@app.command()
def train(
    data_config: Path = typer.Option(
        DEFAULT_DATASET,
        help="ไฟล์ .yaml สำหรับกำหนดโครงสร้างข้อมูล YOLO (train/val/test)"
    ),
    model: str = typer.Option(
        "yolov8n.pt",
        help="โมเดลตั้งต้น (weights) เช่น yolov8n.pt, yolov8s.pt หรือไฟล์ custom",
    ),
    epochs: int = typer.Option(100, help="จำนวนรอบการเทรน"),
    batch: int = typer.Option(32, help="ขนาด batch ต่อครั้ง"),
    imgsz: int = typer.Option(640, help="ขนาดรูปด้านยาว"),
    device: str = typer.Option("auto", help="เลือก GPU/CPU"),
    project_dir: Path = typer.Option(DEFAULT_OUTPUT, help="โฟลเดอร์บันทึกผลลัพธ์"),
    name: str = typer.Option("yolov8-pest", help="ชื่อรอบการทดลอง"),
    patience: int = typer.Option(50, help="จำนวน epoch รอ improvement ก่อน early stop"),
    seed: Optional[int] = typer.Option(None, help="ค่า seed สำหรับ reproducibility"),
) -> None:
    """Train a YOLOv8 detector using the prepared dataset."""
    console.rule("เทรน YOLOv8")
    if not data_config.exists():
        raise typer.BadParameter(f"ไม่พบไฟล์ dataset.yaml ที่ {data_config}")
    project_dir.mkdir(parents=True, exist_ok=True)

    console.log(f"โหลดโมเดลตั้งต้น {model}")
    yolo_model = YOLO(model)

    console.log("เริ่มการเทรนแบบเต็ม")
    results = yolo_model.train(
        data=str(data_config),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=str(project_dir),
        name=name,
    patience=patience,
    save=True,
    exist_ok=True,
    seed=seed,
    )
    console.log("จบการเทรน")
    if results is not None:
        console.print(results)


@app.command()
def export(
    run_dir: Path = typer.Argument(..., help="โฟลเดอร์รอบการเทรนใน models/detector"),
    format: str = typer.Option("onnx", help="รูปแบบไฟล์สำหรับ deployment เช่น onnx, torchscript"),
) -> None:
    """Export trained weights to deployment-friendly formats."""
    weights_path = run_dir / "weights" / "best.pt"
    if not weights_path.exists():
        raise typer.BadParameter(f"ไม่พบไฟล์ {weights_path}")
    console.log(f"โหลดน้ำหนัก {weights_path}")
    model = YOLO(str(weights_path))
    console.log(f"กำลัง export เป็น {format}")
    model.export(format=format)


if __name__ == "__main__":
    app()
