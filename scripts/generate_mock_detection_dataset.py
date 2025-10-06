"""Utility to create a synthetic YOLO-format dataset for smoke testing.

This is intended for environments where the real pest dataset is unavailable but
we still want to exercise the detection training pipeline.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw
import typer
import yaml
from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "detection" / "ai_challenger_yolo"
CLASS_NAMES = [
    "aphid",
    "armyworm",
    "beetle",
    "borer",
    "bug",
    "caterpillar",
    "fly",
    "hopper",
    "mite",
    "weevil",
]

console = Console()
app = typer.Typer(help="Generate a synthetic YOLO dataset for quick testing.")


def _ensure_dirs(base: Path, subsets: Iterable[str]) -> None:
    for subset in subsets:
        images_dir = base / subset / "images"
        labels_dir = base / subset / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)


def _maybe_clear(base: Path) -> None:
    if not base.exists():
        return
    for path in base.rglob("*"):
        if path.is_file():
            path.unlink()
    for path in sorted(base.glob("*"), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def _random_color(rng: random.Random) -> tuple[int, int, int]:
    return tuple(rng.randint(32, 224) for _ in range(3))


def _generate_sample(
    rng: random.Random,
    image_path: Path,
    label_path: Path,
    image_size: tuple[int, int],
    max_boxes: int,
) -> None:
    width, height = image_size
    background = _random_color(rng)
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)

    n_boxes = rng.randint(1, max_boxes)
    lines: list[str] = []
    for _ in range(n_boxes):
        cls_id = rng.randint(0, len(CLASS_NAMES) - 1)
        box_w = rng.uniform(0.1, 0.4) * width
        box_h = rng.uniform(0.1, 0.4) * height
        x1 = rng.uniform(0, width - box_w)
        y1 = rng.uniform(0, height - box_h)
        x2 = x1 + box_w
        y2 = y1 + box_h
        draw.rectangle([x1, y1, x2, y2], outline=_random_color(rng), width=3)
        draw.text((x1 + 4, y1 + 4), CLASS_NAMES[cls_id], fill=(255, 255, 255))

        # Convert to YOLO normalised format
        x_center = ((x1 + x2) / 2) / width
        y_center = ((y1 + y2) / 2) / height
        norm_w = box_w / width
        norm_h = box_h / height
        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

    image.save(image_path)
    label_path.write_text("\n".join(lines), encoding="utf-8")


@app.command()
def main(
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT,
        help="Target directory that should contain train/val/test folders",
    ),
    train_images: int = typer.Option(30, min=1, help="Number of synthetic training images"),
    val_images: int = typer.Option(10, min=1, help="Number of synthetic validation images"),
    test_images: int = typer.Option(10, min=1, help="Number of synthetic test images"),
    image_size: int = typer.Option(640, help="Square image size to generate"),
    max_boxes: int = typer.Option(5, min=1, help="Maximum number of boxes per image"),
    seed: int = typer.Option(1234, help="Random seed for reproducibility"),
    overwrite: bool = typer.Option(
        False,
        help="Remove any existing files under the target directory before generating",
    ),
) -> None:
    """Create a simple synthetic dataset with coloured rectangles and YOLO labels."""
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    if overwrite:
        console.log("Clearing existing dataset contents")
        _maybe_clear(output_dir)

    _ensure_dirs(output_dir, ("train", "val", "test"))
    subsets = {
        "train": train_images,
        "val": val_images,
        "test": test_images,
    }
    size_tuple = (image_size, image_size)

    for subset, count in subsets.items():
        console.log(f"Generating {count} {subset} images")
        images_dir = output_dir / subset / "images"
        labels_dir = output_dir / subset / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(count):
            image_path = images_dir / f"mock_{subset}_{idx:04d}.jpg"
            label_path = labels_dir / f"mock_{subset}_{idx:04d}.txt"
            _generate_sample(rng, image_path, label_path, size_tuple, max_boxes)

    dataset_yaml = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": {idx: name for idx, name in enumerate(CLASS_NAMES)},
    }
    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    console.log("Synthetic dataset ready")


if __name__ == "__main__":
    app()
