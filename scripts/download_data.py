"""Helper CLI to download project datasets."""
from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data import download as download_module

console = Console()
app = typer.Typer(help="เครื่องมือดาวน์โหลดข้อมูลสำหรับโปรเจกต์")


@app.command()
def list() -> None:
    """แสดงรายการชุดข้อมูลที่รองรับ"""
    download_module.list_sources()


@app.command()
def manual(dataset_id: str) -> None:
    """ดูคำแนะนำสำหรับดาวน์โหลดด้วยตนเอง"""
    download_module.manual_instructions(dataset_id)


@app.command()
def file(url: str, output: str | None = None, extract: bool = False) -> None:
    """ดาวน์โหลดไฟล์จากลิงก์สาธารณะ"""
    download_module.download_file(url=url, output=output, extract=extract, extract_dir=None)


@app.command()
def nasa(lat: float, lon: float, start: str, end: str, parameters: str = "T2M,PRECTOTCORR") -> None:
    """ดาวน์โหลดข้อมูลภูมิอากาศจาก NASA POWER"""
    download_module.download_nasa_power(
        latitude=lat,
        longitude=lon,
        start=start,
        end=end,
        parameters=parameters,
    )


@app.command()
def pests_2xlvx(
    overwrite: bool = typer.Option(False, help="ดาวน์โหลดซ้ำและเขียนทับไฟล์เดิม"),
    skip_convert: bool = typer.Option(False, help="ข้ามขั้นตอนแปลงเป็น YOLO"),
) -> None:
    """ดาวน์โหลดชุดข้อมูลตรวจจับศัตรูพืช Pests-2XLVX"""
    download_module.download_pests_2xlvx(overwrite=overwrite, skip_convert=skip_convert)


if __name__ == "__main__":
    app()
