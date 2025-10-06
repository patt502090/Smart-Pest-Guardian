"""Tests for COCO to YOLO conversion utilities in download module."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.data import download


def test_collect_category_mapping_filters_general_class() -> None:
    categories = [
        {"id": 5, "name": "Pests"},
        {"id": 2, "name": "armyworm"},
        {"id": 3, "name": "beetle"},
    ]

    mapping, names = download._collect_category_mapping(categories)

    assert mapping == {2: 0, 3: 1}
    assert names == ["armyworm", "beetle"]


@pytest.mark.parametrize(
    "bbox,width,height,expected",
    [
        ((50.0, 50.0, 100.0, 100.0), 200, 200, "0 0.500000 0.500000 0.500000 0.500000"),
        ((0.0, 0.0, 50.0, 200.0), 200, 200, "0 0.125000 0.500000 0.250000 1.000000"),
    ],
)
def test_convert_coco_split_to_yolo(tmp_path: Path, bbox, width, height, expected: str) -> None:
    split_dir = tmp_path / "train"
    split_dir.mkdir()
    annotations = {
        "images": [
            {"id": 1, "file_name": "image.jpg", "width": width, "height": height},
        ],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 7, "bbox": list(bbox)},
        ],
        "categories": [
            {"id": 7, "name": "armyworm"},
        ],
    }
    (split_dir / "_annotations.coco.json").write_text(json.dumps(annotations), encoding="utf-8")
    (split_dir / "image.jpg").write_bytes(b"fake-image")

    target_base = tmp_path / "out"
    mapping, _ = download._collect_category_mapping(annotations["categories"])
    download._convert_coco_split_to_yolo(
        split_dir=split_dir,
        split_name="train",
        target_base=target_base,
        id_mapping=mapping,
    )

    copied_image = target_base / "train" / "images" / "image.jpg"
    label_file = target_base / "train" / "labels" / "image.txt"

    assert copied_image.exists(), "Image should be copied to YOLO structure"
    assert label_file.exists(), "Label file should be created"
    assert label_file.read_text(encoding="utf-8").strip() == expected


def test_write_dataset_yaml(tmp_path: Path) -> None:
    target_dir = tmp_path / "dataset"
    target_dir.mkdir()
    download._write_dataset_yaml(target_dir, ["armyworm", "beetle"])

    yaml_path = target_dir / "dataset.yaml"
    assert yaml_path.exists()
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert data["nc"] == 2
    assert data["names"] == {0: "armyworm", 1: "beetle"}
    assert data["train"] == "train/images"
    assert data["path"] == str(target_dir.resolve())
