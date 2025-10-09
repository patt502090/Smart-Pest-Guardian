"""Run YOLOv8 predictions on sample balanced images and summarise detections."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO


MODEL_PATH = Path("models/detector/yolov8-pest-rtx4060ti/weights/best.pt")
SOURCE_DIR = Path("data/processed/detection/pests_2xlvx_yolo_balanced/test/images")
OUTPUT_DIR = Path("results/detections/balanced_preview")
NUM_SAMPLES = 12


def run_preview() -> dict[str, Any]:
	if not MODEL_PATH.exists():
		raise FileNotFoundError(f"ไม่พบไฟล์ weight ที่ {MODEL_PATH}")
	if not SOURCE_DIR.exists():
		raise FileNotFoundError(f"ไม่พบโฟลเดอร์ภาพทดสอบที่ {SOURCE_DIR}")

	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	image_paths = sorted(SOURCE_DIR.glob("*.jpg"))[:NUM_SAMPLES]
	if not image_paths:
		raise FileNotFoundError("ไม่พบไฟล์รูป .jpg ใน test/images")

	device = 0 if torch.cuda.is_available() else "cpu"
	model = YOLO(str(MODEL_PATH))

	aggregate_counts: Counter[int] = Counter()
	per_image: list[dict[str, Any]] = []

	predictions = model.predict(
		source=[str(p) for p in image_paths],
		imgsz=640,
		conf=0.25,
		iou=0.45,
		device=device,
		save=True,
		project=str(OUTPUT_DIR.parent),
		name=OUTPUT_DIR.name,
		exist_ok=True,
	)

	for img_path, result in zip(image_paths, predictions):
		img_summary: dict[str, Any] = {
			"image": img_path.name,
			"detections": [],
		}

		if result is None:
			per_image.append(img_summary)
			continue

		name_lookup = result.names
		counts: Counter[int] = Counter()

		if result.boxes is not None and result.boxes.cls is not None:
			cls_ids = result.boxes.cls.to("cpu").tolist()
			for cls in cls_ids:
				cls_int = int(cls)
				counts[cls_int] += 1
				aggregate_counts[cls_int] += 1

		img_summary["detections"] = [
			{"class_id": cid, "class_name": name_lookup.get(cid, f"class_{cid}"), "count": count}
			for cid, count in sorted(counts.items())
		]
		per_image.append(img_summary)

	aggregate_named = {
		result.names.get(cls_id, f"class_{cls_id}"): count for cls_id, count in sorted(aggregate_counts.items())
	}

	payload = {
		"model": str(MODEL_PATH),
		"device": device,
		"num_images": len(image_paths),
		"output_dir": str(OUTPUT_DIR.resolve()),
		"aggregate_counts": aggregate_named,
		"per_image": per_image,
	}

	summary_path = OUTPUT_DIR / "preview_summary.json"
	summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

	print(f"บันทึกผลสรุปไว้ที่ {summary_path}")
	return payload


if __name__ == "__main__":
	summary = run_preview()
	print(json.dumps(summary, ensure_ascii=False, indent=2))
