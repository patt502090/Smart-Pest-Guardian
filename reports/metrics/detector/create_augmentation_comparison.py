from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from PIL import Image, ImageDraw, ImageFont

DATA_DIR = Path("data/processed/detection/pests_2xlvx_yolo_balanced/train/images")
OUTPUT_PATH = Path("reports/metrics/detector/augmentation_comparison.png")
FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/THSarabunNew.ttf"),
    Path("C:/Windows/Fonts/LeelawUI.ttf"),
    Path("C:/Windows/Fonts/Tahoma.ttf"),
    Path("C:/Windows/Fonts/Arial.ttf"),
]


def pick_augmented(base: Path, base_image: Image.Image) -> Optional[Path]:
    stem = base.stem
    base_arr = np.asarray(base_image, dtype=np.float32)
    best_path: Optional[Path] = None
    best_diff = -1.0
    for aug_path in sorted(base.parent.glob(f"{stem}_aug_*")):
        if aug_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        with Image.open(aug_path) as aug_im:
            aug_arr = np.asarray(aug_im.convert("RGB"), dtype=np.float32)
        h = min(base_arr.shape[0], aug_arr.shape[0])
        w = min(base_arr.shape[1], aug_arr.shape[1])
        if h == 0 or w == 0:
            continue
        diff = float(np.mean(np.abs(base_arr[:h, :w] - aug_arr[:h, :w])))
        if diff > best_diff:
            best_diff = diff
            best_path = aug_path
    return best_path


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size)
            except OSError:
                continue
    return ImageFont.load_default()


def compose_image(original: Image.Image, augmented: Image.Image) -> Image.Image:
    max_height = max(original.height, augmented.height)

    def pad(image: Image.Image) -> Image.Image:
        if image.height == max_height:
            return image
        new_im = Image.new("RGB", (image.width, max_height), (255, 255, 255))
        offset = (0, (max_height - image.height) // 2)
        new_im.paste(image, offset)
        return new_im

    original = pad(original)
    augmented = pad(augmented)

    combined = Image.new("RGB", (original.width + augmented.width, max_height + 80), (255, 255, 255))
    combined.paste(original, (0, 40))
    combined.paste(augmented, (original.width, 40))

    draw = ImageDraw.Draw(combined)
    font_title = load_font(32)
    font_caption = load_font(24)

    title = "Before vs After Augmentation"
    title_width = draw.textlength(title, font=font_title)
    draw.text(((combined.width - int(title_width)) // 2, 5), title, fill=(34, 34, 34), font=font_title)

    labels = ["ต้นฉบับ", "หลัง Augmentation"]
    positions = [original.width // 2, original.width + augmented.width // 2]
    for text, x in zip(labels, positions):
        text_width = draw.textlength(text, font=font_caption)
        draw.text((x - int(text_width) // 2, max_height + 45), text, fill=(34, 34, 34), font=font_caption)

    return combined


def main() -> None:
    # pick a representative base image
    base_name = "---_png_jpg.rf.2ff3aeacdfc4e9cae389dc259a7725f8.jpg"
    original_path = DATA_DIR / base_name
    if not original_path.exists():
        raise SystemExit(f"Original image missing: {original_path}")

    original = Image.open(original_path).convert("RGB")
    augmented_path = pick_augmented(original_path, original)
    if not augmented_path:
        raise SystemExit(f"No augmented variant found for {original_path.stem}")

    augmented = Image.open(augmented_path).convert("RGB")

    combined = compose_image(original, augmented)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.save(OUTPUT_PATH)
    print(f"Saved comparison image to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
