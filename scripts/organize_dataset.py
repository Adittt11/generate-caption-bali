"""Move Bali images into dataset/images and number files per category."""

import argparse
import csv
import math
import os
import re
import shutil
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from review_dataset import create_sheet

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
GENERATED_CAPTIONS = {
    "adat_gemini_2caption.csv": "adat_bali",
    "busana_adat_gemini_2caption.csv": "busana_adat_bali",
    "kerajinan_bali_gemini_2caption.csv": "kerajinan_bali",
    "makanan_bali_gemini_2caption.csv": "makanan_tradisional_bali",
    "pura_gemini_2caption.csv": "pura_bali",
    "tari_gemini_2caption.csv": "tari_bali",
}
CAPTION_FIELDS = [
    "image", "caption_1", "caption_2", "word_count_1", "word_count_2",
    "model", "status",
]
REVIEW_FIELDS = [
    "image_path", "folder_category", "source_url", "license", "creator",
    "decision", "actual_category", "notes",
]


def has_image_categories(folder):
    return folder.is_dir() and any(
        child.is_dir() and any(
            image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES
            for image in child.iterdir()
        )
        for child in folder.iterdir()
    )


def natural_key(path):
    parts = re.split(r"(\d+)", path.stem.casefold())
    return tuple(int(part) if part.isdigit() else part for part in parts) + (
        path.suffix.casefold(),
    )


def normalized_suffix(path):
    suffix = path.suffix.lower()
    return ".jpg" if suffix in {".jpg", ".jpeg"} else suffix


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames or [], list(reader)


def write_csv_atomic(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".organize-tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def prepare_csv_updates(image_mapping, source_prefix, categories):
    data_root = PROJECT_ROOT / "data"
    metadata_updates = {}
    metadata_by_image = {}

    for category, old_folder in categories.items():
        metadata_path = PROJECT_ROOT / "dataset" / "images" / category / "metadata.csv"
        if not metadata_path.exists():
            metadata_path = old_folder / "metadata.csv"
        if not metadata_path.exists():
            continue
        fields, rows = read_csv(metadata_path)
        kept = []
        for row in rows:
            old_name = (row.get("image_id") or "").strip()
            metadata_by_image[(category, old_name)] = row
            old_rel = f"{source_prefix}/{category}/{old_name}"
            new_rel = image_mapping.get(old_rel)
            if not new_rel:
                continue
            row["image_id"] = Path(new_rel).name
            kept.append(row)
        metadata_updates[metadata_path] = (fields, kept)

    review_path = data_root / "review" / "review.csv"
    review_update = None
    if review_path.is_file():
        fields, rows = read_csv(review_path)
        kept = []
        represented = set()
        for row in rows:
            old_rel = Path(row.get("image_path", "")).as_posix()
            new_rel = image_mapping.get(old_rel)
            if not new_rel:
                continue
            row["image_path"] = new_rel
            kept.append(row)
            represented.add(new_rel)

        for old_rel, new_rel in image_mapping.items():
            if new_rel in represented:
                continue
            parts = Path(old_rel).parts
            category, old_name = parts[-2], parts[-1]
            metadata = metadata_by_image.get((category, old_name), {})
            kept.append({
                "image_path": new_rel,
                "folder_category": category,
                "source_url": metadata.get("source_url", "") or metadata.get("commons_page", ""),
                "license": metadata.get("license", ""),
                "creator": metadata.get("creator", ""),
                "decision": "review",
                "actual_category": "",
                "notes": "Gambar baru atau belum pernah ditinjau; periksa kategori dan sumbernya.",
            })
        kept.sort(key=lambda row: row["image_path"])
        review_update = (review_path, fields or REVIEW_FIELDS, kept)

    annotation_path = data_root / "annotations_curated.csv"
    annotation_update = None
    if annotation_path.is_file():
        fields, rows = read_csv(annotation_path)
        kept = []
        for row in rows:
            new_rel = image_mapping.get(Path(row.get("image_path", "")).as_posix())
            if new_rel:
                row["image_path"] = new_rel
                kept.append(row)
        annotation_update = (annotation_path, fields, kept)

    caption_updates = []
    captions_root = data_root / "generated_captions"
    for filename, category in GENERATED_CAPTIONS.items():
        path = captions_root / filename
        if not path.is_file():
            continue
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            raw_rows = list(csv.reader(stream))
        if not raw_rows:
            continue
        if raw_rows[0] == CAPTION_FIELDS:
            source_rows = raw_rows[1:]
        elif len(raw_rows[0]) == len(CAPTION_FIELDS):
            # Repair an older export where the first caption row became its header.
            source_rows = raw_rows
        else:
            raise ValueError(f"Format CSV caption tidak dikenali: {path}")
        kept = []
        for values in source_rows:
            row = dict(zip(CAPTION_FIELDS, values))
            old_name = Path(row.get("image", "")).name
            new_rel = image_mapping.get(f"{source_prefix}/{category}/{old_name}")
            if not new_rel:
                continue
            row["image"] = Path(new_rel).name
            kept.append(row)
        caption_updates.append((path, CAPTION_FIELDS, kept))

    return metadata_updates, review_update, annotation_update, caption_updates


def refresh_contact_sheets(image_root, categories, per_page=60):
    review_dir = PROJECT_ROOT / "data" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    expected_sheets = set()
    for category in sorted(categories):
        folder = image_root / category
        images = sorted(
            (p for p in folder.iterdir()
             if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
            key=natural_key,
        )
        page_count = math.ceil(len(images) / per_page)
        for page_index in range(page_count):
            page_number = page_index + 1
            target = review_dir / f"{category}_{page_number:03d}.jpg"
            expected_sheets.add(target.name)
            create_sheet(
                images[page_index * per_page:(page_index + 1) * per_page],
                category, target, page_number, 6, (180, 130),
            )
    for sheet in review_dir.glob("*.jpg"):
        if re.fullmatch(r"[a-z_]+_\d{3}", sheet.stem) and sheet.name not in expected_sheets:
            sheet.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Pindahkan gambar ke dataset/images dan rapikan nomor file per kategori."
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Lakukan pemindahan dan penggantian nama; tanpa opsi ini hanya tampilkan rencana.",
    )
    parser.add_argument("--per-page", type=int, default=60)
    args = parser.parse_args()
    if args.per_page < 1:
        parser.error("--per-page harus positif")

    legacy_root = PROJECT_ROOT / "scraping_image"
    image_root = PROJECT_ROOT / "dataset" / "images"
    if has_image_categories(image_root):
        if has_image_categories(legacy_root):
            raise SystemExit(
                "Ada gambar di scraping_image/ dan dataset/images/. "
                "Pindahkan atau satukan salah satunya terlebih dahulu untuk menghindari timpa nama."
            )
        source_root, source_prefix, migrate = image_root, "dataset/images", False
    elif has_image_categories(legacy_root):
        source_root, source_prefix, migrate = legacy_root, "scraping_image", True
    else:
        raise SystemExit("Tidak menemukan folder kategori berisi gambar.")

    source_folders = {
        folder.name: folder for folder in sorted(source_root.iterdir())
        if folder.is_dir() and any(
            p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES for p in folder.iterdir()
        )
    }
    mapping = {}
    plans = {}
    for category, folder in source_folders.items():
        images = sorted(
            (p for p in folder.iterdir()
             if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
            key=natural_key,
        )
        for index, source in enumerate(images, start=1):
            suffix = normalized_suffix(source)
            new_rel = Path("dataset/images") / category / f"{index:04d}{suffix}"
            old_rel = source.relative_to(PROJECT_ROOT).as_posix()
            mapping[old_rel] = new_rel.as_posix()
        plans[category] = images

    if not mapping:
        raise SystemExit("Tidak ada gambar untuk ditata.")
    if len(set(mapping.values())) != len(mapping):
        raise SystemExit("Ada nama file tujuan duplikat; tidak ada perubahan dilakukan.")

    metadata_updates, review_update, annotation_update, caption_updates = prepare_csv_updates(
        mapping, source_prefix, source_folders
    )
    print(f"Folder gambar tujuan: {image_root.relative_to(PROJECT_ROOT)}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY RUN'} | Gambar: {len(mapping)}")
    for category, images in plans.items():
        print(f"{category}: {len(images)} gambar")
    if not args.apply:
        print("Belum ada file yang diubah. Jalankan ulang dengan --apply untuk menerapkan rencana.")
        return

    image_root.mkdir(parents=True, exist_ok=True)
    if migrate:
        for category, source_folder in source_folders.items():
            target_folder = image_root / category
            if target_folder.exists():
                if any(target_folder.iterdir()):
                    raise SystemExit(f"Folder tujuan sudah berisi file: {target_folder}")
                target_folder.rmdir()
            shutil.move(str(source_folder), str(target_folder))

    # Rename through temporary filenames so existing numbered files never collide.
    temporary_moves = []
    token = uuid.uuid4().hex[:10]
    for old_rel, new_rel in mapping.items():
        source = PROJECT_ROOT / old_rel
        if migrate:
            source = image_root / Path(old_rel).relative_to("scraping_image")
        target = PROJECT_ROOT / new_rel
        temporary = source.with_name(f".rename-{token}-{len(temporary_moves):05d}.tmp")
        os.replace(source, temporary)
        temporary_moves.append((temporary, target))
    for temporary, target in temporary_moves:
        os.replace(temporary, target)

    # Preserve an earlier rename manifest by composing its destinations.
    manifest_path = PROJECT_ROOT / "data" / "filename_rename_manifest.csv"
    prior_manifest = []
    if manifest_path.is_file():
        _, prior_manifest = read_csv(manifest_path)
    prior_sources = {row.get("new_path", "") for row in prior_manifest}
    manifest_rows = []
    for row in prior_manifest:
        previous_destination = row.get("new_path", "")
        row["new_path"] = mapping.get(previous_destination, previous_destination)
        manifest_rows.append(row)
    for old_rel, new_rel in mapping.items():
        if old_rel not in prior_sources:
            manifest_rows.append({"old_path": old_rel, "new_path": new_rel})
    write_csv_atomic(manifest_path, ["old_path", "new_path"], manifest_rows)

    for path, (fields, rows) in metadata_updates.items():
        if migrate:
            category = path.parent.name
            path = image_root / category / "metadata.csv"
        write_csv_atomic(path, fields, rows)
    if review_update:
        write_csv_atomic(*review_update)
    if annotation_update:
        write_csv_atomic(*annotation_update)
    for update in caption_updates:
        write_csv_atomic(*update)

    # Remove operating-system folder metadata and refresh thumbnails without replacing decisions.
    for category in source_folders:
        for hidden in (image_root / category).glob(".DS_Store"):
            hidden.unlink()
    refresh_contact_sheets(image_root, source_folders, args.per_page)
    print(f"Selesai. Pemetaan nama lama-baru tersimpan di {manifest_path.relative_to(PROJECT_ROOT)}")
    if review_update:
        print("Gambar yang belum ada di review.csv ditambahkan dengan decision=review.")


if __name__ == "__main__":
    main()
