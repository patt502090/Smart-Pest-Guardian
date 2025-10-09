from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd

DATASET_DIR = Path('data/processed/detection/pests_2xlvx_yolo_balanced')
DATASET_CONFIG = DATASET_DIR / 'dataset.yaml'
TEST_LABELS = DATASET_DIR / 'test/labels'
OUTPUT_DIR = Path('reports/metrics/detector')
OUTPUT_PNG = OUTPUT_DIR / 'test_split_counts.png'
OUTPUT_CSV = OUTPUT_DIR / 'test_split_counts.csv'


def ensure_font() -> None:
    for font_name in ("Tahoma", "Leelawadee UI", "Angsana New", "Sarabun"):
        if any(font_name == font.name for font in fm.fontManager.ttflist):
            plt.rcParams['font.family'] = font_name
            return


def load_class_names() -> dict[int, str]:
    import yaml

    if not DATASET_CONFIG.exists():
        raise FileNotFoundError(f"dataset.yaml not found at {DATASET_CONFIG}")
    with DATASET_CONFIG.open('r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    names_field = config.get('names')
    if isinstance(names_field, dict):
        return {int(k): v for k, v in names_field.items()}
    if isinstance(names_field, list):
        return {idx: name for idx, name in enumerate(names_field)}
    raise ValueError('Unexpected format for names in dataset.yaml')


def compute_counts() -> Counter[int]:
    if not TEST_LABELS.exists():
        raise FileNotFoundError(f"Test labels directory is missing: {TEST_LABELS}")
    counts: Counter[int] = Counter()
    for label_path in TEST_LABELS.glob('*.txt'):
        text = label_path.read_text(encoding='utf-8').strip()
        if not text:
            continue
        for line in text.splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                cls = int(float(parts[0]))
            except ValueError:
                continue
            counts[cls] += 1
    return counts


def main() -> None:
    ensure_font()
    names = load_class_names()
    counts = compute_counts()

    # Include classes even if zero to make gaps obvious
    series = pd.Series({names.get(idx, f'class_{idx}'): counts.get(idx, 0) for idx in sorted(names)})

    df = series.sort_values(ascending=True).to_frame(name='จำนวนภาพ (test)')
    df.to_csv(OUTPUT_CSV, encoding='utf-8-sig')

    fig, ax = plt.subplots(figsize=(12, 9))
    df.plot(kind='barh', ax=ax, legend=False, color='#60a5fa', edgecolor='black')
    ax.set_xlabel('จำนวนภาพใน test set')
    ax.set_title('จำนวนภาพต่อคลาสใน Test Set หลังแยกจาก Train/Val', fontsize=16)
    for i, (count) in enumerate(df['จำนวนภาพ (test)']):
        ax.text(count + 0.5, i, str(count), va='center', fontsize=11)
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=300)
    plt.close(fig)
    print(json.dumps({'png': str(OUTPUT_PNG), 'csv': str(OUTPUT_CSV), 'total_images': int(df['จำนวนภาพ (test)'].sum())}, ensure_ascii=False))


if __name__ == '__main__':
    main()
