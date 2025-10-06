"""Training utilities for YOLOv8-based pest detection."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console
import torch
from ultralytics import YOLO

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - matplotlib backend issues
    plt = None

console = Console()
app = typer.Typer(help="เทรนโมเดลตรวจจับศัตรูพืชด้วย YOLOv8")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "processed" / "detection" / "pests_2xlvx_yolo" / "dataset.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "detector"
DEFAULT_METRICS_DIR = PROJECT_ROOT / "reports" / "metrics" / "detector"


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
    device: str = typer.Option("auto", help="เลือก GPU/CPU (auto = เลือกเอง)"),
    project_dir: Path = typer.Option(DEFAULT_OUTPUT, help="โฟลเดอร์บันทึกผลลัพธ์"),
    name: str = typer.Option("yolov8-pest", help="ชื่อรอบการทดลอง"),
    patience: int = typer.Option(50, help="จำนวน epoch รอ improvement ก่อน early stop"),
    seed: Optional[int] = typer.Option(None, help="ค่า seed สำหรับ reproducibility"),
    metrics_dir: Path = typer.Option(
        DEFAULT_METRICS_DIR, help="โฟลเดอร์บันทึกผลลัพธ์ metrics และ summary"
    ),
    save_curves: bool = typer.Option(True, help="สร้างกราฟสรุปผลการเทรน"),
) -> None:
    """Train a YOLOv8 detector using the prepared dataset."""
    console.rule("เทรน YOLOv8")
    if not data_config.exists():
        raise typer.BadParameter(f"ไม่พบไฟล์ dataset.yaml ที่ {data_config}")
    project_dir.mkdir(parents=True, exist_ok=True)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        console.log(f"เลือกอุปกรณ์ที่ใช้เทรน: {device}")

    console.log(f"โหลดโมเดลตั้งต้น {model}")
    yolo_model = YOLO(model)

    if seed is None:
        seed = 0

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
    trainer = getattr(yolo_model, "trainer", None)
    run_dir = Path(trainer.save_dir) if trainer and getattr(trainer, "save_dir", None) else project_dir / name
    metrics_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        output_csv = metrics_dir / f"{name}_{timestamp}_results.csv"
        df.to_csv(output_csv, index=False)
        console.log(f"บันทึก metrics CSV ที่ {output_csv}")

        summary = df.iloc[-1].to_dict()
        summary_path = metrics_dir / f"{name}_{timestamp}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        console.log(f"สรุปผลลัพธ์ล่าสุดที่ {summary_path}")

        if save_curves:
            if plt is None:
                console.print("[yellow]ไม่สามารถสร้างกราฟได้ (matplotlib ไม่พร้อมใช้งาน)[/yellow]")
            else:
                metrics_columns = [col for col in df.columns if col.startswith("metrics/")]
                loss_columns = [col for col in df.columns if col.startswith("train/") or col.startswith("val/")]
                if metrics_columns or loss_columns:
                    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                    if metrics_columns:
                        for column in metrics_columns:
                            axes[0].plot(df["epoch"], df[column], label=column.split("/", 1)[-1])
                        axes[0].set_title("Metrics")
                        axes[0].set_ylabel("Value")
                        axes[0].legend(loc="best")
                        axes[0].grid(True, alpha=0.3)
                    else:
                        axes[0].axis("off")

                    if loss_columns:
                        for column in loss_columns:
                            axes[1].plot(df["epoch"], df[column], label=column)
                        axes[1].set_title("Loss")
                        axes[1].set_xlabel("Epoch")
                        axes[1].set_ylabel("Loss")
                        axes[1].legend(loc="best")
                        axes[1].grid(True, alpha=0.3)
                    else:
                        axes[1].axis("off")

                    fig.tight_layout()
                    plot_path = metrics_dir / f"{name}_{timestamp}_curves.png"
                    fig.savefig(plot_path, dpi=150)
                    plt.close(fig)
                    console.log(f"บันทึกกราฟที่ {plot_path}")
    else:
        console.print("[yellow]ไม่พบไฟล์ results.csv ในโฟลเดอร์การเทรน[/yellow]")

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
