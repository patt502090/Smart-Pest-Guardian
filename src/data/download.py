"""Utility script for downloading public datasets used in the project.

This module provides a Typer-powered CLI with helper functions to fetch and
prepare archives. For datasets that require authentication (e.g. Kaggle),
the script will display step-by-step instructions instead of attempting an
unauthorized download.
"""
from __future__ import annotations

import tarfile
import zipfile
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional

import requests
import typer
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

console = Console()
app = typer.Typer(help="ดาวน์โหลดและจัดการชุดข้อมูลสำหรับระบบศัตรูพืชอัจฉริยะ")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "detection"

PESTS_2XLVX_ID = "pests_2xlvx"
PESTS_2XLVX_ARCHIVE_URL = "https://huggingface.co/datasets/Francesco/pests-2xlvx/resolve/main/dataset.tar.gz"
PESTS_2XLVX_ARCHIVE_NAME = "pests-2xlvx.tar.gz"
PESTS_2XLVX_RAW_DIR = DATA_DIR / "pests-2xlvx"
PESTS_2XLVX_PROCESSED_DIR = PROCESSED_DIR / "pests_2xlvx_yolo"

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
    PESTS_2XLVX_ID: {
        "description": "Pests-2XLVX (ODinW RF100) - real pest detection set from Hugging Face",
        "url": PESTS_2XLVX_ARCHIVE_URL,
        "requires_auth": False,
        "notes": "ใช้คำสั่ง pests-2xlvx เพื่อดาวน์โหลดและแปลงเป็น YOLO",
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


def _extract_tar_subset(
    archive_path: Path,
    subset_token: str,
    destination: Path,
    overwrite: bool,
) -> None:
    if destination.exists() and overwrite:
        console.log(f"ลบไฟล์เก่าใน {destination}")
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            member_parts = Path(member.name).parts
            if subset_token not in member_parts:
                continue
            idx = member_parts.index(subset_token)
            new_relative = Path(*member_parts[idx + 1 :])
            target_path = destination / new_relative
            target_path.parent.mkdir(parents=True, exist_ok=True)
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            with extracted as source, target_path.open("wb") as target_file:
                shutil.copyfileobj(source, target_file)


def _collect_category_mapping(categories: Iterable[dict]) -> tuple[Dict[int, int], list[str]]:
    id_to_name = {
        category["id"]: category["name"]
        for category in categories
        if category["name"].lower() != "pests"
    }
    sorted_ids = sorted(id_to_name)
    id_to_index = {category_id: idx for idx, category_id in enumerate(sorted_ids)}
    ordered_names = [id_to_name[category_id] for category_id in sorted_ids]
    return id_to_index, ordered_names


def _convert_coco_split_to_yolo(
    split_dir: Path,
    split_name: str,
    target_base: Path,
    id_mapping: Dict[int, int],
) -> None:
    annotations_file = split_dir / "_annotations.coco.json"
    if not annotations_file.exists():
        raise typer.BadParameter(f"ไม่พบ annotation สำหรับ {split_name}: {annotations_file}")

    data = json.loads(annotations_file.read_text(encoding="utf-8"))
    images_by_id = {image["id"]: image for image in data.get("images", [])}
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in data.get("annotations", []):
        category_id = annotation.get("category_id")
        if category_id not in id_mapping:
            continue
        annotations_by_image[annotation["image_id"]].append(annotation)

    images_dir = target_base / split_name / "images"
    labels_dir = target_base / split_name / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for image_id, image_meta in images_by_id.items():
        filename = image_meta["file_name"]
        width = image_meta["width"]
        height = image_meta["height"]
        source_path = split_dir / filename
        target_path = images_dir / filename
        if not source_path.exists():
            console.log(f"[yellow]ข้าม {source_path} เพราะไม่พบไฟล์ภาพ[/yellow]")
            continue
        shutil.copy2(source_path, target_path)

        label_lines = []
        for annotation in annotations_by_image.get(image_id, []):
            bbox = annotation["bbox"]
            if len(bbox) != 4:
                continue
            x, y, box_w, box_h = bbox
            x_center = (x + box_w / 2) / width
            y_center = (y + box_h / 2) / height
            norm_w = box_w / width
            norm_h = box_h / height
            yolo_class = id_mapping[annotation["category_id"]]
            label_lines.append(
                f"{yolo_class} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
            )

        label_path = labels_dir / f"{Path(filename).stem}.txt"
        label_path.write_text("\n".join(label_lines), encoding="utf-8")


def _write_dataset_yaml(target_dir: Path, class_names: Iterable[str]) -> None:
    import yaml

    names_list = list(class_names)
    content = {
        "path": str(target_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(names_list),
        "names": {idx: name for idx, name in enumerate(names_list)},
    }
    yaml_path = target_dir / "dataset.yaml"
    yaml_path.write_text(
        yaml.safe_dump(content, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


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


@app.command()
def download_pests_2xlvx(
    overwrite: bool = typer.Option(False, help="ดาวน์โหลดซ้ำและเขียนทับไฟล์เดิม"),
    skip_convert: bool = typer.Option(False, help="ข้ามขั้นตอนแปลงเป็น YOLO"),
) -> None:
    """ดาวน์โหลดชุดข้อมูล Pests-2XLVX และเตรียมเป็นฟอร์แมต YOLO"""
    _ensure_dir(DATA_DIR)
    archive_path = DATA_DIR / PESTS_2XLVX_ARCHIVE_NAME
    if overwrite or not archive_path.exists():
        console.log("เริ่มดาวน์โหลดชุดข้อมูล Pests-2XLVX")
        _download_file(PESTS_2XLVX_ARCHIVE_URL, archive_path)
    else:
        console.log("พบไฟล์ดาวน์โหลดเดิมแล้ว ข้ามขั้นตอนดาวน์โหลด")

    console.log("แตกไฟล์ไปยัง data/raw/pests-2xlvx")
    _extract_tar_subset(
        archive_path=archive_path,
        subset_token="pests-2xlvx",
        destination=PESTS_2XLVX_RAW_DIR,
        overwrite=overwrite,
    )

    if skip_convert:
        console.log("ข้ามขั้นตอนแปลง YOLO ตามที่กำหนด")
        return

    console.log("แปลง annotation (COCO) เป็น YOLO")
    class_names: list[str] | None = None
    for split in ("train", "valid", "test"):
        split_dir = PESTS_2XLVX_RAW_DIR / split
        if not split_dir.exists():
            console.log(f"[yellow]ไม่พบโฟลเดอร์ {split_dir} - ข้าม[/yellow]")
            continue
        annotations_file = split_dir / "_annotations.coco.json"
        if not annotations_file.exists():
            console.log(f"[yellow]ไม่พบไฟล์ {annotations_file} - ข้าม[/yellow]")
            continue
        coco_data = json.loads(annotations_file.read_text(encoding="utf-8"))
        if class_names is None:
            id_mapping, class_names = _collect_category_mapping(coco_data.get("categories", []))
        else:
            id_mapping, _ = _collect_category_mapping(coco_data.get("categories", []))
        _convert_coco_split_to_yolo(
            split_dir=split_dir,
            split_name="val" if split == "valid" else split,
            target_base=PESTS_2XLVX_PROCESSED_DIR,
            id_mapping=id_mapping,
        )

    if class_names:
        _write_dataset_yaml(PESTS_2XLVX_PROCESSED_DIR, class_names)
        console.log(f"ชุดข้อมูลพร้อมใช้งานที่ {PESTS_2XLVX_PROCESSED_DIR}")
    else:
        console.log("[yellow]ไม่สามารถสร้างไฟล์ dataset.yaml ได้ เนื่องจากไม่พบ categories[/yellow]")


if __name__ == "__main__":
    app()
