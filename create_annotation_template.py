import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    parser = argparse.ArgumentParser(
        description="Buat template anotasi manual dari folder gambar budaya Bali."
    )
    parser.add_argument("--image-root", default="scraping_image")
    parser.add_argument("--output", default="data/annotations.csv")
    parser.add_argument("--captions-per-image", type=int, default=2)
    args = parser.parse_args()

    image_root = (ROOT / args.image_root).resolve()
    output_path = (ROOT / args.output).resolve()
    if args.captions_per_image < 1:
        raise SystemExit("--captions-per-image harus minimal 1.")
    if not image_root.is_dir():
        raise SystemExit(f"Folder gambar tidak ditemukan: {image_root}")
    if output_path.exists():
        raise SystemExit(f"File tidak ditimpa karena sudah ada: {output_path}")

    images = sorted(
        path for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise SystemExit(f"Tidak ada gambar di {image_root}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["image_path", "caption", "category", "reviewed"],
        )
        writer.writeheader()
        for image_path in images:
            relative_path = image_path.relative_to(ROOT).as_posix()
            category = image_path.parent.name
            for _ in range(args.captions_per_image):
                writer.writerow({
                    "image_path": relative_path,
                    "caption": "",
                    "category": category,
                    "reviewed": "no",
                })

    print(f"Template dibuat: {output_path}")
    print(f"Gambar: {len(images)} | Baris caption: {len(images) * args.captions_per_image}")
    print("Isi caption dan ubah reviewed menjadi yes setelah diperiksa manusia.")


if __name__ == "__main__":
    main()
