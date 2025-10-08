"""Dataset balancing and augmentation utilities for YOLOv8 pest detection."""
from __future__ import annotations

import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(help="สร้างชุดข้อมูลใหม่ที่ปรับสมดุลคลาสด้วย data augmentation")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "processed" / "detection" / "pests_2xlvx_yolo"
DEFAULT_TARGET = PROJECT_ROOT / "data" / "processed" / "detection" / "pests_2xlvx_yolo_balanced"
DEFAULT_REPORTS = PROJECT_ROOT / "reports" / "metrics" / "detector"


@dataclass
class YoloSample:
    image_path: Path
    label_path: Path
    class_ids: Sequence[int]


def load_yolo_label(label_path: Path) -> Tuple[List[List[float]], List[int]]:
    """Load YOLO-format label file."""
    boxes: List[List[float]] = []
    classes: List[int] = []
    if not label_path.exists():
        return boxes, classes

    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                class_id = int(float(parts[0]))
                bbox = [float(v) for v in parts[1:]]
            except ValueError:
                continue
            classes.append(class_id)
            boxes.append(bbox)
    return boxes, classes


def save_yolo_label(label_path: Path, boxes: Sequence[Sequence[float]], classes: Sequence[int]) -> None:
    """Persist YOLO-format labels."""
    lines = []
    for cls, bbox in zip(classes, boxes):
        values = [str(cls), *[f"{v:.6f}" for v in bbox]]
        lines.append(" ".join(values))
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def gather_samples(train_dir: Path, rare_class_ids: Iterable[int]) -> Dict[int, List[YoloSample]]:
    """Collect training samples that contain the specified classes."""
    labels_dir = train_dir / "labels"
    images_dir = train_dir / "images"
    class_to_samples: Dict[int, List[YoloSample]] = defaultdict(list)

    for label_path in labels_dir.glob("*.txt"):
        boxes, classes = load_yolo_label(label_path)
        if not classes:
            continue
        image_path = images_dir / f"{label_path.stem}.jpg"
        if not image_path.exists():
            # try png/jpeg alternatives
            for ext in (".png", ".jpeg", ".JPG", ".PNG"):
                candidate = images_dir / f"{label_path.stem}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
        if not image_path.exists():
            continue

        unique_classes = set(classes)
        for class_id in unique_classes & set(rare_class_ids):
            class_to_samples[class_id].append(YoloSample(image_path=image_path, label_path=label_path, class_ids=classes))

    return class_to_samples


def build_augmentation_pipeline() -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(p=0.4),
            A.RandomBrightnessContrast(p=0.4),
            A.HueSaturationValue(p=0.3),
            A.Blur(blur_limit=3, p=0.2),
            A.RandomScale(scale_limit=0.2, p=0.3),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.2),
    )


def compute_class_counts(train_dir: Path, class_count: int) -> Counter:
    labels_dir = train_dir / "labels"
    counts: Counter = Counter({idx: 0 for idx in range(class_count)})
    for label_path in labels_dir.glob("*.txt"):
        _, classes = load_yolo_label(label_path)
        for class_id in classes:
            counts[class_id] += 1
    return counts


def render_count_table(counts: Counter, class_names: Dict[int, str]) -> Table:
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Class ID", justify="right")
    table.add_column("Name")
    table.add_column("Count", justify="right")
    for class_id, count in sorted(counts.items(), key=lambda item: item[1]):
        table.add_row(str(class_id), class_names.get(class_id, f"class_{class_id}"), str(count))
    return table


def maybe_copy_dataset(source: Path, target: Path) -> None:
    if target.exists():
        console.log(f"พบโฟลเดอร์ที่ {target} แล้ว จะใช้ข้อมูลเดิม")
        return
    console.log(f"กำลังก๊อปปี้ชุดข้อมูลไปยัง {target}")
    shutil.copytree(source, target)


def augment_class(
    class_id: int,
    samples: List[YoloSample],
    target_dir: Path,
    target_count: int,
    class_names: Dict[int, str],
    pipeline: A.Compose,
) -> int:
    labels_dir = target_dir / "train" / "labels"
    images_dir = target_dir / "train" / "images"
    current_counts = compute_class_counts(target_dir / "train", len(class_names))
    start_count = current_counts[class_id]
    needed = max(0, target_count - start_count)
    if needed == 0:
        console.log(f"คลาส {class_names[class_id]} มี {start_count} ตัวอย่าง เพียงพอแล้ว")
        return 0

    console.log(f"เพิ่มข้อมูลคลาส {class_names[class_id]}: {start_count} -> {target_count} (ต้องเพิ่ม {needed})")
    rng = random.Random(20251009 + class_id)
    generated = 0
    sample_pool = samples or []
    if not sample_pool:
        console.log(f"[yellow]ไม่มีตัวอย่างต้นฉบับของคลาส {class_names[class_id]} ให้ augment[/yellow]")
        return 0

    while generated < needed:
        sample = rng.choice(sample_pool)
        image = cv2.imread(str(sample.image_path))
        if image is None:
            console.log(f"[yellow]ไม่สามารถเปิดรูป {sample.image_path}[/yellow]")
            continue
        boxes, classes = load_yolo_label(sample.label_path)
        if not boxes:
            continue

        attempts = 0
        success = False
        while attempts < 10 and not success:
            transformed = pipeline(image=image, bboxes=boxes, class_labels=classes)
            new_boxes = transformed["bboxes"]
            new_classes = transformed["class_labels"]
            if not new_boxes:
                attempts += 1
                continue
            # ensure at least one bbox of target class remains
            if all(cls != class_id for cls in new_classes):
                attempts += 1
                continue

            aug_name = f"{sample.image_path.stem}_aug_{class_id}_{generated:03d}"
            out_image_path = images_dir / f"{aug_name}.jpg"
            out_label_path = labels_dir / f"{aug_name}.txt"
            cv2.imwrite(str(out_image_path), transformed["image"])
            save_yolo_label(out_label_path, new_boxes, new_classes)
            generated += 1
            success = True
            if generated % 10 == 0 or generated == needed:
                console.log(f"  -> สร้างรูปเพิ่มแล้ว {generated}/{needed}")
        if not success:
            console.log(f"[yellow]สร้าง augmentation ไม่สำเร็จหลังพยายามหลายครั้งสำหรับ {sample.image_path}[/yellow]")
            break

    return generated


def save_summary(report_dir: Path, class_names: Dict[int, str], before: Counter, after: Counter, target_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "target_dataset": str(target_dir.resolve()),
        "before": {class_names[k]: before[k] for k in before},
        "after": {class_names[k]: after[k] for k in after},
    }
    summary_path = report_dir / "dataset_balance_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    console.log(f"บันทึกสรุปการปรับสมดุลไว้ที่ {summary_path}")


@app.command()
def balance(
    source: Path = typer.Option(DEFAULT_SOURCE, help="โฟลเดอร์ชุดข้อมูล YOLO เดิม"),
    target: Path = typer.Option(DEFAULT_TARGET, help="โฟลเดอร์สำหรับชุดข้อมูลใหม่หลังปรับสมดุล"),
    class_count: int = typer.Option(28, help="จำนวนคลาสทั้งหมด"),
    target_per_class: int = typer.Option(60, help="จำนวนอย่างน้อยต่อคลาสหลัง augment"),
    reports_dir: Path = typer.Option(DEFAULT_REPORTS, help="โฟลเดอร์บันทึกสรุป"),
) -> None:
    """Balance the dataset by augmenting under-represented classes."""
    console.rule("ปรับสมดุลชุดข้อมูล")
    if not source.exists():
        raise typer.BadParameter(f"ไม่พบชุดข้อมูลที่ {source}")

    config_path = source / "dataset.yaml"
    if not config_path.exists():
        raise typer.BadParameter("ไม่พบ dataset.yaml ในต้นทาง")

    # lazy import using yaml without enforcing dependency at runtime if unavailable
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise typer.BadParameter("ต้องติดตั้ง PyYAML ก่อนใช้งาน") from exc

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names: Dict[int, str] = {int(k): v for k, v in data["names"].items()}
    maybe_copy_dataset(source, target)

    target_config_path = target / "dataset.yaml"
    if not target_config_path.exists():
        data_copy = dict(data)
        data_copy["path"] = str(target.resolve())
        with target_config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data_copy, f, allow_unicode=True)
        console.log(f"เขียน dataset.yaml ใหม่ที่ {target_config_path}")

    before_counts = compute_class_counts(target / "train", class_count)
    console.log("จำนวนตัวอย่างก่อน augment")
    console.print(render_count_table(before_counts, names))

    rare_classes = [idx for idx, count in before_counts.items() if count < target_per_class]
    if not rare_classes:
        console.print("[green]ทุกคลาสมีตัวอย่างถึงเป้าหมายแล้ว[/green]")
        return

    console.log(f"พบคลาสที่ต้อง augment: {[names[idx] for idx in rare_classes]}")
    samples_map = gather_samples(target / "train", rare_classes)
    pipeline = build_augmentation_pipeline()

    total_generated = 0
    for class_id in rare_classes:
        generated = augment_class(class_id, samples_map.get(class_id, []), target, target_per_class, names, pipeline)
        total_generated += generated

    after_counts = compute_class_counts(target / "train", class_count)
    console.log("จำนวนตัวอย่างหลัง augment")
    console.print(render_count_table(after_counts, names))
    console.log(f"สร้างรูปเพิ่มทั้งหมด {total_generated}")
    save_summary(reports_dir, names, before_counts, after_counts, target)
    console.rule("เสร็จสิ้น")


if __name__ == "__main__":
    app()
