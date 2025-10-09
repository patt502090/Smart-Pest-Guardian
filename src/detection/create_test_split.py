"""Utility to expand or create a dedicated test split for YOLO datasets."""
from __future__ import annotations

import math
import random
import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table


app = typer.Typer(help="จัดสรรข้อมูลไปยัง test split ตามสัดส่วนที่ต้องการ")
console = Console()


def _count_images(folder: Path) -> int:
    return sum(1 for _ in folder.glob("*.jpg"))


def _render_counts(root: Path) -> Table:
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Split")
    table.add_column("Images", justify="right")
    table.add_row("train", str(_count_images(root / "train" / "images")))
    table.add_row("val", str(_count_images(root / "val" / "images")))
    table.add_row("test", str(_count_images(root / "test" / "images")))
    return table


def _move_sample(image_path: Path, src_root: Path, dst_root: Path) -> bool:
    label_path = src_root / "labels" / (image_path.stem + ".txt")
    dst_image = dst_root / "images" / image_path.name
    dst_label = dst_root / "labels" / (image_path.stem + ".txt")

    dst_image.parent.mkdir(parents=True, exist_ok=True)
    dst_label.parent.mkdir(parents=True, exist_ok=True)

    if not label_path.exists():
        console.log(f"[yellow]ข้ามไฟล์ {image_path.name} เพราะหา label ไม่เจอ[/yellow]")
        return False

    shutil.move(str(image_path), str(dst_image))
    shutil.move(str(label_path), str(dst_label))
    return True


@app.command()
def ensure_test_split(
    root: Path = typer.Argument(
        Path("data/processed/detection/pests_2xlvx_yolo_balanced"),
        help="โฟลเดอร์หลักของชุดข้อมูล YOLO",
    ),
    test_fraction: float = typer.Option(0.12, min=0.05, max=0.3, help="สัดส่วนภาพที่ควรอยู่ใน test"),
    seed: int = typer.Option(20251009, help="seed สำหรับสุ่ม"),
) -> None:
    """Move samples from train/val into test until reaching the desired fraction."""
    console.rule("เตรียม test split")
    rng = random.Random(seed)

    train_images = sorted((root / "train" / "images").glob("*.jpg"))
    val_images = sorted((root / "val" / "images").glob("*.jpg"))
    test_images = sorted((root / "test" / "images").glob("*.jpg"))

    total_images = len(train_images) + len(val_images) + len(test_images)
    desired_test = math.ceil(total_images * test_fraction)
    current_test = len(test_images)
    needed = desired_test - current_test

    console.log(f"จำนวนภาพทั้งหมด {total_images}")
    console.log(f"จำนวน test ปัจจุบัน {current_test}")
    console.log(f"ต้องการ test อย่างน้อย {desired_test}")

    if needed <= 0:
        console.print("[green]จำนวน test ปัจจุบันเพียงพอแล้ว[/green]")
        console.print(_render_counts(root))
        return

    console.log(f"ต้องย้ายเพิ่ม {needed} ภาพไปยัง test")

    candidate_pool = val_images + train_images
    if len(candidate_pool) < needed:
        raise typer.BadParameter("จำนวนภาพใน train/val ไม่พอสำหรับย้ายไป test")

    rng.shuffle(candidate_pool)
    moved = 0
    for image_path in candidate_pool:
        src_root = root / ("val" if image_path in val_images else "train")
        if _move_sample(image_path, src_root, root / "test"):
            moved += 1
        if moved >= needed:
            break

    console.log(f"ย้ายภาพสำเร็จ {moved}/{needed}")
    console.print(_render_counts(root))
    console.rule("เสร็จสิ้น")


if __name__ == "__main__":
    app()
