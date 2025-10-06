"""Explainability helpers for detection and forecasting modules."""
from __future__ import annotations

import shutil
from pathlib import Path

import typer
from rich.console import Console
from ultralytics import YOLO

console = Console()
app = typer.Typer(help="สร้างภาพ heatmap เพื่ออธิบายผลลัพธ์")


@app.command()
def visualize(
    weights: Path = typer.Argument(..., help="ไฟล์น้ำหนัก YOLO เช่น best.pt"),
    image_path: Path = typer.Argument(..., help="ภาพที่จะสร้าง heatmap"),
    output_path: Path = typer.Option(Path("results/explainability/heatmap.jpg"), help="ไฟล์ผลลัพธ์"),
    conf: float = typer.Option(0.25, help="threshold ความมั่นใจ"),
    imgsz: int = typer.Option(640, help="ขนาดภาพ"),
) -> None:
    """Create a feature-visualization heatmap using Ultralytics built-in hooks.

    ฟังก์ชันนี้ใช้ `visualize=True` ของ Ultralytics YOLO เพื่อลงสีบริเวณที่โมเดลให้ความสำคัญ
    โดยไม่พึ่งพาไลบรารีภายนอกอย่าง pytorch-grad-cam ซึ่งยังไม่รองรับ Python 3.12
    """

    if not image_path.exists():
        raise typer.BadParameter(f"ไม่พบภาพ {image_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    results = model.predict(
        source=str(image_path),
        conf=conf,
        imgsz=imgsz,
        save=True,
        visualize=True,
        project=str(output_path.parent),
        name="_tmp_heatmap",
        exist_ok=True,
    )

    if not results or not getattr(results[0], "visualize_path", None):
        raise RuntimeError("ไม่สามารถสร้าง heatmap ได้ โปรดลองลดค่า conf หรือเลือกภาพที่มีวัตถุ")

    shutil.copy(results[0].visualize_path, output_path)
    console.log(f"สร้าง heatmap ที่ {output_path}")


if __name__ == "__main__":
    app()
