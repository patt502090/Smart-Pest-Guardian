"""Inference helpers for pest detection models."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer
from rich.console import Console
from ultralytics import YOLO

console = Console()
app = typer.Typer(help="รันการตรวจจับศัตรูพืชกับภาพหรือวิดีโอ")


@app.command()
def run_images(
    weights: Path = typer.Argument(..., help="พาธไปยังไฟล์ weights เช่น best.pt"),
    image_dir: Path = typer.Argument(..., help="โฟลเดอร์ภาพที่ต้องการตรวจจับ"),
    output_dir: Path = typer.Option(Path("results/detections"), help="โฟลเดอร์สำหรับผลลัพธ์"),
    conf: float = typer.Option(0.25, help="threshold ความมั่นใจ"),
    save_txt: bool = typer.Option(True, help="บันทึกผลเป็น txt"),
    save_csv: bool = typer.Option(True, help="บันทึกผลเป็น CSV"),
) -> None:
    """รันการตรวจจับกับชุดภาพ"""
    if not image_dir.exists():
        raise typer.BadParameter(f"ไม่พบโฟลเดอร์ภาพ {image_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    console.log("เริ่มตรวจจับ...")
    results = model.predict(source=str(image_dir), conf=conf, save=save_txt, project=str(output_dir), exist_ok=True)

    if save_csv:
        console.log("รวบรวมผลลัพธ์เป็น CSV")
        records = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes else []
            scores = result.boxes.conf.cpu().numpy() if result.boxes else []
            classes = result.boxes.cls.cpu().numpy() if result.boxes else []
            for box, score, cls in zip(boxes, scores, classes):
                records.append(
                    {
                        "image_path": result.path,
                        "x1": box[0],
                        "y1": box[1],
                        "x2": box[2],
                        "y2": box[3],
                        "confidence": float(score),
                        "class_id": int(cls),
                    }
                )
        if records:
            df = pd.DataFrame.from_records(records)
            csv_path = output_dir / "detections.csv"
            df.to_csv(csv_path, index=False)
            console.log(f"บันทึก CSV ที่ {csv_path}")


@app.command()
def summarize(
    detections_csv: Path = typer.Argument(..., help="ไฟล์ผลลัพธ์ CSV จาก run_images"),
    class_map_json: Optional[Path] = typer.Option(None, help="ไฟล์ map class_id -> ชื่อศัตรูพืช"),
) -> None:
    """สรุปจำนวนการตรวจจับต่อคลาส"""
    if not detections_csv.exists():
        raise typer.BadParameter(f"ไม่พบไฟล์ {detections_csv}")
    df = pd.read_csv(detections_csv)
    summary = df.groupby("class_id").agg(
        detections=("confidence", "count"),
        avg_confidence=("confidence", "mean"),
    )
    if class_map_json and class_map_json.exists():
        class_map = json.loads(class_map_json.read_text(encoding="utf-8"))
        summary.index = summary.index.map(lambda cid: class_map.get(str(cid), cid))
    console.print(summary)


if __name__ == "__main__":
    app()
