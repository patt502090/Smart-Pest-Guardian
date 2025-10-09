from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd

ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "dataset_balance_summary.json"
OUTPUT_PATH = ROOT / "class_counts.png"


def main() -> None:
    # try to set a font with Thai glyph support (Tahoma is available on Windows by default)
    for font_name in ("Tahoma", "Leelawadee UI", "Angsana New"):
        if any(font_name == font.name for font in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = font_name
            break

    if not SUMMARY_PATH.exists():
        raise SystemExit(f"Summary file not found: {SUMMARY_PATH}")

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    before = pd.Series(summary["before"], name="ก่อน augment")
    after = pd.Series(summary["after"], name="หลัง augment")

    df = pd.concat([before, after], axis=1)
    df = df.sort_values(by="ก่อน augment")

    fig, ax = plt.subplots(figsize=(12, 9))
    df.plot(kind="barh", ax=ax, color=["#d9534f", "#5bc0de"], edgecolor="black")

    ax.set_title("จำนวนภาพต่อคลาสก่อนและหลังการทำ Data Augmentation", fontsize=16)
    ax.set_xlabel("จำนวนภาพ")
    ax.set_ylabel("ชื่อคลาส")
    ax.legend(loc="lower right")
    plt.tight_layout()

    fig.savefig(OUTPUT_PATH, dpi=300)
    plt.close(fig)
    print(f"Saved class count chart to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
