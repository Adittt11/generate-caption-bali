"""Create contact sheets and a CSV checklist for manually auditing image folders."""

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError


ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
COLUMNS = [
    "image_path", "folder_category", "source_url", "license", "creator",
    "decision", "actual_category", "notes",
]


def load_metadata(folder):
    metadata_path = folder / "metadata.csv"
    if not metadata_path.is_file():
        return {}
    with metadata_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        return {row.get("image_id", ""): row for row in csv.DictReader(csv_file)}


def create_sheet(image_paths, category, destination, page_number, columns, thumb_size):
    rows = (len(image_paths) + columns - 1) // columns
    label_height = 28
    sheet = Image.new(
        "RGB",
        (columns * thumb_size[0], rows * (thumb_size[1] + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, image_path in enumerate(image_paths):
        x = (index % columns) * thumb_size[0]
        y = (index // columns) * (thumb_size[1] + label_height)
        try:
            with Image.open(image_path) as opened:
                thumbnail = ImageOps.contain(opened.convert("RGB"), thumb_size)
            offset_x = x + (thumb_size[0] - thumbnail.width) // 2
            offset_y = y + (thumb_size[1] - thumbnail.height) // 2
            sheet.paste(thumbnail, (offset_x, offset_y))
        except (UnidentifiedImageError, OSError):
            draw.rectangle((x, y, x + thumb_size[0] - 1, y + thumb_size[1] - 1), fill="#eeeeee")
            draw.text((x + 8, y + 8), "GAMBAR RUSAK", fill="red", font=font)
        draw.text(
            (x + 4, y + thumb_size[1] + 3),
            image_path.name[:28],
            fill="black",
            font=font,
        )
    sheet.save(destination, quality=88)


def main():
    parser = argparse.ArgumentParser(
        description="Buat contact sheet dan CSV untuk memeriksa gambar per kategori."
    )
    parser.add_argument("--image-root", default="dataset/images")
    parser.add_argument("--output-dir", default="data/review")
    parser.add_argument("--per-page", type=int, default=60)
    args = parser.parse_args()
    if args.per_page < 1:
        parser.error("--per-page harus positif")

    image_root = (ROOT / args.image_root).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    review_csv = output_dir / "review.csv"
    if review_csv.exists():
        raise SystemExit(
            f"File review sudah ada dan tidak ditimpa: {review_csv}. "
            "Pilih --output-dir baru agar keputusan yang sudah diisi tetap aman."
        )

    review_rows = []
    sheet_count = 0
    for folder in sorted(path for path in image_root.iterdir() if path.is_dir()):
        images = sorted(
            path for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        metadata = load_metadata(folder)
        for start in range(0, len(images), args.per_page):
            page_images = images[start:start + args.per_page]
            page_number = start // args.per_page + 1
            sheet_path = output_dir / f"{folder.name}_{page_number:03d}.jpg"
            create_sheet(page_images, folder.name, sheet_path, page_number, 6, (180, 130))
            sheet_count += 1

        for image_path in images:
            row = metadata.get(image_path.name, {})
            review_rows.append({
                "image_path": image_path.relative_to(ROOT).as_posix(),
                "folder_category": folder.name,
                "source_url": row.get("source_url", ""),
                "license": row.get("license", ""),
                "creator": row.get("creator", ""),
                "decision": "",
                "actual_category": "",
                "notes": "",
            })

    if not review_rows:
        raise SystemExit(f"Tidak ada gambar di {image_root}")
    with review_csv.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(review_rows)
    print(f"Gambar: {len(review_rows)} | Contact sheet: {sheet_count}")
    print(f"Buka contact sheet di: {output_dir}")
    print(f"Isi kolom decision di: {review_csv}")
    print("Gunakan keep, reject, atau review; hanya keep yang akan masuk template anotasi.")


if __name__ == "__main__":
    main()
