from collections import defaultdict
from pathlib import Path

labels = Path("data/processed/detection/pests_2xlvx_yolo_balanced/test/labels")
class_to_files = defaultdict(list)
for path in labels.glob("*.txt"):
    for line in path.read_text().splitlines():
        if not line:
            continue
        cls = int(float(line.split()[0]))
        class_to_files[cls].append(path.name)

for target in (4, 0, 5, 13):
    files = class_to_files.get(target, [])[:5]
    print(f"class {target}: {files}")
