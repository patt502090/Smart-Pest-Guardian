from pathlib import Path
import cv2
import numpy as np

DATA = Path('data/processed/detection/pests_2xlvx_yolo_balanced/train/images')
results = []
for aug in DATA.glob('*_aug_*'):
    base_name = aug.name.split('_aug_')[0] + '.jpg'
    base = DATA / base_name
    if not base.exists():
        continue
    img_base = cv2.imread(str(base))
    img_aug = cv2.imread(str(aug))
    if img_base is None or img_aug is None:
        continue
    h = min(img_base.shape[0], img_aug.shape[0])
    w = min(img_base.shape[1], img_aug.shape[1])
    base_crop = img_base[:h, :w]
    aug_crop = img_aug[:h, :w]
    diff = np.mean(np.abs(base_crop.astype(float) - aug_crop.astype(float)))
    results.append((diff, base.name, aug.name))

results.sort(reverse=True)
for diff, base_name, aug_name in results[:30]:
    print(f"diff={diff:.1f} base={base_name} aug={aug_name}")
