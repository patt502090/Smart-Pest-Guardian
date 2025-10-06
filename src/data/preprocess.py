"""Data preparation utilities for detection and forecasting tasks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import typer
from rich.console import Console

console = Console()
app = typer.Typer(help="เตรียมข้อมูลสำหรับงานตรวจจับและพยากรณ์")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "detection").mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "forecasting").mkdir(parents=True, exist_ok=True)


@app.command()
def convert_ai_challenger_to_yolo(
    source_dir: Path = typer.Option(
        RAW_DIR / "AI_Challenger_Pest", exists=False, help="โฟลเดอร์ที่มีไฟล์ annotation ของ AI Challenger"
    ),
    output_dir: Path = typer.Option(
        PROCESSED_DIR / "detection" / "ai_challenger_yolo", help="โฟลเดอร์ผลลัพธ์ในรูปแบบ YOLO"
    ),
    train_ratio: float = typer.Option(0.7, help="สัดส่วน train"),
    val_ratio: float = typer.Option(0.2, help="สัดส่วน validation"),
) -> None:
    """ตัวอย่างฟังก์ชันแปลง annotation ของ AI Challenger ไปเป็นรูปแบบ YOLOv8"""
    console.rule("เริ่มการแปลงข้อมูล AI Challenger")
    _ensure_dirs()
    if not source_dir.exists():
        raise typer.BadParameter("ไม่พบโฟลเดอร์ AI Challenger โปรดตรวจสอบพาธ")

    images_dir = source_dir / "images"
    annotations_file = source_dir / "annotations.json"
    if not annotations_file.exists():
        raise typer.BadParameter("ต้องมีไฟล์ annotations.json จาก AI Challenger")

    with annotations_file.open("r", encoding="utf-8") as f:
        annotations = json.load(f)

    image_to_ann: Dict[str, List[Dict]] = {}
    for item in annotations["annotations"]:
        image_to_ann.setdefault(str(item["image_id"]), []).append(item)

    splits: Dict[str, List[Tuple[str, List[Dict]]]] = {"train": [], "val": [], "test": []}
    image_items = list(annotations["images"])
    total_images = len(image_items)
    train_cutoff = int(total_images * train_ratio)
    val_cutoff = int(total_images * (train_ratio + val_ratio))

    for idx, image_info in enumerate(image_items):
        file_name = image_info["file_name"]
        image_id = str(image_info["id"])
        anns = image_to_ann.get(image_id, [])
        if idx < train_cutoff:
            target_split = "train"
        elif idx < val_cutoff:
            target_split = "val"
        else:
            target_split = "test"
        splits[target_split].append((file_name, anns, image_info["width"], image_info["height"]))

    for split, items in splits.items():
        split_dir = output_dir / split
        labels_dir = split_dir / "labels"
        images_out_dir = split_dir / "images"
        labels_dir.mkdir(parents=True, exist_ok=True)
        images_out_dir.mkdir(parents=True, exist_ok=True)

        for file_name, anns, width, height in items:
            src_image_path = images_dir / file_name
            dst_image_path = images_out_dir / file_name
            if not src_image_path.exists():
                console.print(f"[yellow]ข้าม {file_name} เพราะไม่พบไฟล์ภาพ[/yellow]")
                continue
            dst_image_path.write_bytes(src_image_path.read_bytes())

            label_lines = []
            for ann in anns:
                category_id = ann["category_id"]
                bbox = ann["bbox"]  # [x, y, width, height]
                x_center = bbox[0] + bbox[2] / 2
                y_center = bbox[1] + bbox[3] / 2
                label_lines.append(
                    f"{category_id} {x_center / width:.6f} {y_center / height:.6f} {bbox[2] / width:.6f} {bbox[3] / height:.6f}"
                )
            (labels_dir / (Path(file_name).stem + ".txt")).write_text("\n".join(label_lines), encoding="utf-8")
    console.log(f"บันทึกผลลัพธ์ไว้ที่ {output_dir}")


@app.command()
def aggregate_detection_to_timeseries(
    detections_csv: Path = typer.Argument(..., help="ไฟล์ CSV ที่บันทึกผลตรวจจับระดับภาพ/เฟรม"),
    output_csv: Path = typer.Option(
        PROCESSED_DIR / "forecasting" / "pest_incidents_daily.csv", help="ปลายทาง time series"
    ),
    datetime_column: str = typer.Option("captured_at", help="คอลัมน์เวลา"),
    location_column: str = typer.Option("location_id", help="คอลัมน์พื้นที่"),
    pest_column: str = typer.Option("pest_class", help="คอลัมน์ชนิดศัตรูพืช"),
    score_column: str = typer.Option("confidence", help="คอลัมน์ความมั่นใจ"),
    min_confidence: float = typer.Option(0.25, help="กรองค่าความมั่นใจต่ำสุด"),
) -> None:
    """สรุปผลตรวจจับเป็น time series รายวันเพื่อใช้พยากรณ์"""
    console.rule("สรุปผลตรวจจับเป็น Time Series")
    _ensure_dirs()

    def _unwrap(value):
        if isinstance(value, typer.models.OptionInfo):
            return value.default
        return value

    datetime_column = _unwrap(datetime_column)
    location_column = _unwrap(location_column)
    pest_column = _unwrap(pest_column)
    score_column = _unwrap(score_column)
    min_confidence = _unwrap(min_confidence)

    df = pd.read_csv(detections_csv, parse_dates=[datetime_column])
    df = df.loc[df[score_column] >= min_confidence]
    df["date"] = df[datetime_column].dt.date

    group_cols = ["date", location_column, pest_column]
    summary = (
        df.groupby(group_cols)
        .agg(
            detections_count=(score_column, "count"),
            mean_confidence=(score_column, "mean"),
        )
        .reset_index()
    )

    pivot_counts = summary.pivot_table(
        index=["date", location_column],
        columns=pest_column,
        values="detections_count",
        fill_value=0,
    )
    def _slugify(value: str) -> str:
        import re

        slug = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip().lower())
        return slug.strip("_")

    pivot_counts.columns = [f"count_{_slugify(col)}" for col in pivot_counts.columns]

    overall_summary = (
        df.groupby(["date", location_column])
        .agg(
            count_total=(score_column, "count"),
            mean_confidence=(score_column, "mean"),
        )
        .reset_index()
    )

    merged = overall_summary.merge(
        pivot_counts.reset_index(),
        on=["date", location_column],
        how="left",
    )
    merged.fillna(0, inplace=True)

    count_columns = sorted({col for col in merged.columns if col.startswith("count_")})
    merged["date"] = pd.to_datetime(merged["date"])
    filled_frames = []
    for loc, group in merged.groupby(location_column):
        group = group.sort_values("date")
        full_range = pd.date_range(group["date"].min(), group["date"].max(), freq="D")
        reindexed = group.set_index("date").reindex(full_range)
        reindexed[count_columns] = reindexed[count_columns].fillna(0)
        reindexed["mean_confidence"] = reindexed["mean_confidence"].fillna(0.0)
        reindexed[location_column] = loc
        reindexed.reset_index(inplace=True)
        reindexed.rename(columns={"index": "date"}, inplace=True)
        filled_frames.append(reindexed)

    merged = pd.concat(filled_frames, ignore_index=True)
    merged["date"] = merged["date"].dt.date

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    console.log(f"บันทึก time series ที่ {output_csv}")


if __name__ == "__main__":
    app()
