import yaml
from collections import Counter
from pathlib import Path

root = Path("data/processed/detection/pests_2xlvx_yolo")
with open(root / "dataset.yaml", "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

names = data["names"]
counts = Counter()
for split in ["train", "val"]:
    labels_dir = root / split / "labels"
    if not labels_dir.exists():
        continue
    for path in labels_dir.glob("*.txt"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    class_id = int(float(parts[0]))
                except Exception:
                    continue
                counts[class_id] += 1

sorted_counts = sorted(counts.items(), key=lambda x: x[1])
for class_id, count in sorted_counts:
    name = names.get(class_id, f"class_{class_id}")
    print(f"{class_id:2d} | {name:30s} | {count:4d}")

rare = [class_id for class_id, count in sorted_counts if count < 15]
print("\nClasses below 15 instances:", len(rare))
print(rare)
