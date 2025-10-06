"""Utility script for downloading public datasets used in the project.

This module provides a Typer-powered CLI with helper functions to fetch and
prepare archives. For datasets that require authentication (e.g. Kaggle),
the script will display step-by-step instructions instead of attempting an
unauthorized download.
"""
from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path
from typing import Optional

import requests
import typer
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

console = Console()
app = typer.Typer(help="ดาวน์โหลดและจัดการชุดข้อมูลสำหรับระบบศัตรูพืชอัจฉริยะ")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

DATASETS = {
    "ai_challenger_pest": {
        "description": "AI Challenger 2018 - Pests in crops (requires manual download)",
        "url": "https://github.com/AIChallenger/AI_Challenger_2018/tree/master/PestDisease_Agricultural",
        "requires_auth": True,
        "notes": "สมัครบัญชี/ดาวน์โหลดจากหน้า GitHub/เว็บไซต์ทางการ แล้ววางไฟล์ zip ลงใน data/raw",
    },
    "plantvillage": {
        "description": "PlantVillage dataset (Kaggle) - leaf images with disease labels",
        "url": "https://www.kaggle.com/datasets/emmarex/plantdisease",
        "requires_auth": True,
        "notes": "ต้องใช้ Kaggle API หรือดาวน์โหลดด้วยตนเอง แล้ววางไฟล์ใน data/raw/plantvillage",
    },
    "nasa_power": {
        "description": "NASA POWER agroclimatology daily data (accessible via API)",
        "url": "https://power.larc.nasa.gov/api/",
        "requires_auth": False,
        "notes": "ใช้คำสั่ง --download-nasa เพื่อดึงข้อมูลพื้นที่เฉพาะ",
    },
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _download_file(url: str, destination: Path) -> Path:
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with destination.open("wb") as file, tqdm(
        total=total, unit="B", unit_scale=True, desc=f"ดาวน์โหลด {destination.name}"
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))
    return destination


def _extract_archive(archive_path: Path, target_dir: Path) -> None:
    console.log(f"กำลังแตกไฟล์ {archive_path} ไปยัง {target_dir}")
    if archive_path.suffix in {".zip"}:
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)
    elif archive_path.suffix in {".gz", ".tgz"} or archive_path.suffixes[-2:] == [".tar", ".gz"]:
        with tarfile.open(archive_path, "r:gz") as tar_ref:
            tar_ref.extractall(target_dir)
    else:
        raise typer.BadParameter(f"ไม่รู้วิธีแตกไฟล์: {archive_path.suffix}")


@app.command()
def list_sources() -> None:
    """แสดงรายการชุดข้อมูลที่รองรับ"""
    table = Table(title="Datasets ที่รองรับ", box=None)
    table.add_column("รหัส", style="cyan", no_wrap=True)
    table.add_column("คำอธิบาย")
    table.add_column("ต้องล็อกอิน")
    table.add_column("หมายเหตุ")
    for key, meta in DATASETS.items():
        table.add_row(key, meta["description"], "✅" if meta["requires_auth"] else "❌", meta["notes"])
    console.print(table)


@app.command()
def manual_instructions(dataset_id: str = typer.Argument(..., help="รหัสชุดข้อมูล")) -> None:
    """แสดงคำแนะนำการดาวน์โหลดแบบ manual สำหรับชุดข้อมูลที่ต้องใช้บัญชี"""
    dataset = DATASETS.get(dataset_id)
    if not dataset:
        raise typer.BadParameter(f"ไม่พบชุดข้อมูลรหัส '{dataset_id}'")
    if not dataset["requires_auth"]:
        console.print("[green]ชุดข้อมูลนี้สามารถดาวน์โหลดผ่านสคริปต์ได้โดยตรง[/green]")
        console.print(f"URL: {dataset['url']}")
        return
    console.rule(f"คำแนะนำสำหรับ {dataset_id}")
    console.print(dataset["description"], style="bold")
    console.print(f"1. เข้าไปที่: [link={dataset['url']}]หน้าดาวน์โหลด[/link]")
    console.print("2. ล็อกอิน/ลงทะเบียนตามที่แพลตฟอร์มกำหนด")
    console.print("3. ดาวน์โหลดไฟล์ .zip หรือ .tar.gz")
    console.print("4. นำไฟล์ที่ได้มาวางไว้ในโฟลเดอร์ data/raw/ แล้วใช้คำสั่ง extract-archive เพื่อแตกไฟล์")


@app.command()
def download_file(
    url: str = typer.Argument(..., help="ลิงก์ไฟล์ที่สามารถดาวน์โหลดโดยตรง"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="พาธไฟล์ปลายทาง"),
    extract: bool = typer.Option(False, "--extract", help="แตกไฟล์หลังดาวน์โหลด"),
    extract_dir: Optional[Path] = typer.Option(None, help="โฟลเดอร์สำหรับแตกไฟล์"),
) -> None:
    """ดาวน์โหลดไฟล์จากลิงก์สาธารณะและ (อาจ) แตกไฟล์"""
    _ensure_dir(DATA_DIR)
    destination = output or (DATA_DIR / Path(url).name)
    console.log(f"เริ่มดาวน์โหลดจาก {url}")
    file_path = _download_file(url, destination)
    console.log(f"ดาวน์โหลดเสร็จแล้ว: {file_path}")
    if extract:
        target_dir = extract_dir or DATA_DIR / file_path.stem
        _ensure_dir(target_dir)
        _extract_archive(file_path, target_dir)
        console.log(f"แตกไฟล์เรียบร้อยที่ {target_dir}")


@app.command()
def extract_archive(
    archive_path: Path = typer.Argument(..., help="ไฟล์ .zip/.tar.gz ที่อยู่ในเครื่อง"),
    target_dir: Optional[Path] = typer.Option(None, "--to", help="ปลายทางที่ต้องการให้แตกไฟล์"),
) -> None:
    """แตกไฟล์ที่ดาวน์โหลดมาแล้ว"""
    if not archive_path.exists():
        raise typer.BadParameter(f"ไม่พบไฟล์ {archive_path}")
    destination = target_dir or DATA_DIR / archive_path.stem
    _ensure_dir(destination)
    _extract_archive(archive_path, destination)
    console.log("แตกไฟล์สำเร็จ")


@app.command()
def download_nasa_power(
    latitude: float = typer.Option(..., help="ละติจูดศูนย์กลางพื้นที่"),
    longitude: float = typer.Option(..., help="ลองจิจูดศูนย์กลางพื้นที่"),
    start: str = typer.Option(..., help="วันที่เริ่มต้นรูปแบบ YYYYMMDD"),
    end: str = typer.Option(..., help="วันที่สิ้นสุดรูปแบบ YYYYMMDD"),
    parameters: str = typer.Option(
        "T2M,PRECTOTCORR,RH2M,ALLSKY_SFC_SW_DWN",
        help="ตัวแปรภูมิอากาศที่ต้องการ (คั่นด้วยเครื่องหมายจุลภาค)",
    ),
    community: str = typer.Option("AG", help="กลุ่มผู้ใช้งาน (ค่าแนะนำ: AG)")
) -> None:
    """ดึงข้อมูลภูมิอากาศรายวันจาก NASA POWER API"""
    _ensure_dir(DATA_DIR / "climate")
    output_file = DATA_DIR / "climate" / f"nasa_power_{latitude}_{longitude}_{start}_{end}.json"
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    query = {
        "start": start,
        "end": end,
        "latitude": latitude,
        "longitude": longitude,
        "parameters": parameters,
        "community": community,
        "format": "JSON",
    }
    console.log("เรียก API NASA POWER...")
    response = requests.get(base_url, params=query, timeout=60)
    response.raise_for_status()
    output_file.write_text(response.text, encoding="utf-8")
    console.log(f"บันทึกข้อมูลที่ {output_file}")


if __name__ == "__main__":
    app()
