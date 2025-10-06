"""Inference helpers for pest detection models."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

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
    location_id: str = typer.Option("site-1", help="รหัสพื้นที่สำหรับการรวมข้อมูล"),
    captured_at: Optional[str] = typer.Option(
        None,
        help="กำหนดเวลาเก็บภาพแบบคงที่หรือจุดเริ่ม (ISO8601) ขึ้นกับ timestamp-mode",
    ),
    timestamp_mode: str = typer.Option(
        "file",
        help="วิธีตั้งค่าเวลา: file = ใช้เวลาจากไฟล์, fixed = ใช้ค่าเดียว, sequence = เวลาต่อเนื่อง",
    ),
    sequence_step_minutes: int = typer.Option(1440, help="ช่วงเวลาห่างกันในโหมด sequence (นาที)"),
) -> None:
    """รันการตรวจจับกับชุดภาพ"""
    if not image_dir.exists():
        raise typer.BadParameter(f"ไม่พบโฟลเดอร์ภาพ {image_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    console.log("เริ่มตรวจจับ...")
    results = model.predict(source=str(image_dir), conf=conf, save=save_txt, project=str(output_dir), exist_ok=True)

    raw_names = getattr(model, "names", {})
    if isinstance(raw_names, dict):
        class_map = {int(k): v for k, v in raw_names.items()}
    else:
        class_map = {idx: name for idx, name in enumerate(raw_names)}

    timestamp_mode = timestamp_mode.lower()
    if timestamp_mode not in {"file", "fixed", "sequence"}:
        raise typer.BadParameter("timestamp-mode ต้องเป็น file, fixed หรือ sequence")

    fixed_timestamp = None
    if captured_at:
        try:
            fixed_timestamp = datetime.fromisoformat(captured_at)
        except ValueError as err:
            raise typer.BadParameter("captured_at ต้องอยู่ในรูปแบบ ISO8601 เช่น 2024-01-01T00:00:00") from err

    if timestamp_mode == "fixed" and fixed_timestamp is None:
        raise typer.BadParameter("ระบุ captured-at เมื่อใช้ timestamp-mode=fixed")

    if timestamp_mode == "sequence":
        base_timestamp = fixed_timestamp or datetime.utcnow()
        step = timedelta(minutes=sequence_step_minutes)
        sequence_index = 0

    if save_csv:
        console.log("รวบรวมผลลัพธ์เป็น CSV")
        records = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes else []
            scores = result.boxes.conf.cpu().numpy() if result.boxes else []
            classes = result.boxes.cls.cpu().numpy() if result.boxes else []
            image_path = Path(result.path)
            if timestamp_mode == "fixed":
                timestamp = fixed_timestamp  # type: ignore[assignment]
            elif timestamp_mode == "sequence":
                timestamp = base_timestamp + step * sequence_index
                sequence_index += 1
            else:
                try:
                    timestamp = datetime.fromtimestamp(image_path.stat().st_mtime)
                except FileNotFoundError:
                    timestamp = datetime.utcnow()
            for box, score, cls in zip(boxes, scores, classes):
                class_id = int(cls)
                records.append(
                    {
                        "image_path": str(image_path),
                        "x1": box[0],
                        "y1": box[1],
                        "x2": box[2],
                        "y2": box[3],
                        "confidence": float(score),
                        "class_id": class_id,
                        "pest_class": class_map.get(class_id, str(class_id)),
                        "captured_at": timestamp.isoformat(),
                        "location_id": location_id,
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
